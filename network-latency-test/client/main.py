"""Cart-side latency benchmark client.

Runs on the Jetson Thor (or any Linux box for testing). Keeps a single QUIC
connection to the deployed Modal server warm, continuously fires fixed-size
requests at `REQUEST_RATE_HZ`, and exposes a tiny FastAPI UI on
`http://localhost:5000` so an operator can start/stop recording.

Wire protocol matches `server/main.py`:

    Client -> Server: [u64 client_seq][u64 client_send_ns][u32 payload_len][payload]
    Server -> Client: [u64 client_seq][u64 client_send_ns]
                      [u64 server_recv_ns][u64 server_send_ns]
                      [u32 payload_len][payload]

CSV columns per response (or anomaly event):

    seq, t_send_ns, t_server_recv_ns, t_server_send_ns, t_recv_ns,
    rtt_ms, uplink_ms_est, downlink_ms_est, server_proc_ms,
    payload_size, event

Uplink/downlink estimates assume `time.time_ns()` on both ends is
NTP-synchronised — accuracy is bounded by the residual clock skew, which is
usually 1-10 ms on a `chronyd`-managed host. See the README.
"""

from __future__ import annotations

import csv
import logging
import os
import secrets
import signal
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

LOG = logging.getLogger("latency-bench-client")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

APP_NAME = os.environ.get("MODAL_APP_NAME", "latency-bench")
COORD_DICT_NAME = os.environ.get("MODAL_COORD_DICT", "latency-bench-coord")
SERVER_FUNCTION = os.environ.get("MODAL_SERVER_FUNCTION", "run_server")
MODAL_QUIC_ENDPOINT = os.environ.get("MODAL_QUIC_ENDPOINT", f"modal://{APP_NAME}/{SERVER_FUNCTION}")

PAYLOAD_SIZE_BYTES = int(os.environ.get("PAYLOAD_SIZE_BYTES", "500000"))
REQUEST_RATE_HZ = float(os.environ.get("REQUEST_RATE_HZ", "5"))
TIMEOUT_MS = int(os.environ.get("TIMEOUT_MS", "2000"))
RECORDING_DIR = Path(os.environ.get("RECORDING_DIR", "./recordings")).resolve()
CLIENT_LOCAL_PORT = int(os.environ.get("CLIENT_LOCAL_PORT", "5556"))
UI_HOST = os.environ.get("UI_HOST", "127.0.0.1")
UI_PORT = int(os.environ.get("UI_PORT", "5000"))
FLUSH_EVERY_N = int(os.environ.get("FLUSH_EVERY_N", "50"))
RTT_WINDOW_SIZE = int(os.environ.get("RTT_WINDOW_SIZE", "1024"))

CLIENT_HDR = struct.Struct("<QQI")
SERVER_HDR = struct.Struct("<QQQQI")

CSV_COLUMNS = [
    "seq",
    "t_send_ns",
    "t_server_recv_ns",
    "t_server_send_ns",
    "t_recv_ns",
    "rtt_ms",
    "uplink_ms_est",
    "downlink_ms_est",
    "server_proc_ms",
    "payload_size",
    "event",
]


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


@dataclass
class RecordingSummary:
    label: str
    path: str
    started_at: float
    stopped_at: float
    total_requests: int
    responses: int
    stalls: int
    drops: int
    rtt_min_ms: Optional[float]
    rtt_p50_ms: Optional[float]
    rtt_p90_ms: Optional[float]
    rtt_p99_ms: Optional[float]
    rtt_max_ms: Optional[float]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.stopped_at - self.started_at)

    @property
    def stall_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.stalls / self.total_requests


class Recorder:
    """Thread-safe CSV recorder. Buffers writes and flushes every N rows."""

    def __init__(self, recording_dir: Path) -> None:
        self.recording_dir = recording_dir
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = None
        self._writer: Optional[csv.writer] = None
        self._buffered = 0
        self._label: Optional[str] = None
        self._path: Optional[Path] = None
        self._started_at: float = 0.0
        self._sent_baseline = 0
        self._recv_baseline = 0
        self._stall_baseline = 0
        self._drop_baseline = 0
        self._rtt_samples: list[float] = []

    @property
    def active(self) -> bool:
        return self._file is not None

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def path(self) -> Optional[str]:
        return str(self._path) if self._path else None

    def start(self, label: str, tracker: "StatsTracker") -> str:
        with self._lock:
            if self._file is not None:
                raise RuntimeError("recording already active")
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label) or "run"
            ts = int(time.time())
            path = self.recording_dir / f"{safe}_{ts}.csv"
            f = path.open("w", newline="", buffering=1)
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            self._file = f
            self._writer = writer
            self._buffered = 0
            self._label = safe
            self._path = path
            self._started_at = time.time()
            snap = tracker.snapshot()
            self._sent_baseline = snap["sent"]
            self._recv_baseline = snap["received"]
            self._stall_baseline = snap["stalls"]
            self._drop_baseline = snap["drops"]
            self._rtt_samples = []
            LOG.info("recording start label=%s path=%s", safe, path)
            return str(path)

    def write_response(
        self,
        *,
        seq: int,
        t_send_ns: int,
        t_server_recv_ns: int,
        t_server_send_ns: int,
        t_recv_ns: int,
        rtt_ms: float,
        uplink_ms_est: float,
        downlink_ms_est: float,
        server_proc_ms: float,
        payload_size: int,
    ) -> None:
        with self._lock:
            if self._writer is None:
                return
            self._writer.writerow(
                [
                    seq,
                    t_send_ns,
                    t_server_recv_ns,
                    t_server_send_ns,
                    t_recv_ns,
                    f"{rtt_ms:.6f}",
                    f"{uplink_ms_est:.6f}",
                    f"{downlink_ms_est:.6f}",
                    f"{server_proc_ms:.6f}",
                    payload_size,
                    "",
                ]
            )
            self._rtt_samples.append(rtt_ms)
            self._buffered += 1
            if self._buffered >= FLUSH_EVERY_N:
                self._file.flush()
                self._buffered = 0

    def write_event(
        self,
        event: str,
        *,
        seq: int = 0,
        t_send_ns: int = 0,
        payload_size: int = 0,
    ) -> None:
        with self._lock:
            if self._writer is None:
                return
            now_ns = time.time_ns()
            self._writer.writerow(
                [seq, t_send_ns, 0, 0, now_ns, "", "", "", "", payload_size, event]
            )
            self._buffered += 1
            self._file.flush()
            self._buffered = 0
            LOG.info("recording event=%s seq=%d", event, seq)

    def stop(self, tracker: "StatsTracker") -> Optional[RecordingSummary]:
        with self._lock:
            if self._file is None:
                return None
            try:
                self._file.flush()
            finally:
                self._file.close()
            stopped_at = time.time()
            path = str(self._path)
            label = self._label or ""
            snap = tracker.snapshot()
            samples = sorted(self._rtt_samples)
            n = len(samples)

            def pct(p: float) -> Optional[float]:
                if n == 0:
                    return None
                idx = min(n - 1, max(0, int(round(p * (n - 1)))))
                return samples[idx]

            summary = RecordingSummary(
                label=label,
                path=path,
                started_at=self._started_at,
                stopped_at=stopped_at,
                total_requests=snap["sent"] - self._sent_baseline,
                responses=snap["received"] - self._recv_baseline,
                stalls=snap["stalls"] - self._stall_baseline,
                drops=snap["drops"] - self._drop_baseline,
                rtt_min_ms=samples[0] if n else None,
                rtt_p50_ms=pct(0.50),
                rtt_p90_ms=pct(0.90),
                rtt_p99_ms=pct(0.99),
                rtt_max_ms=samples[-1] if n else None,
            )
            self._file = None
            self._writer = None
            self._path = None
            self._label = None
            self._buffered = 0
            self._rtt_samples = []
            LOG.info("recording stop summary=%s", summary)
            return summary


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def _percentile(sorted_vals, p: float):
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = min(n - 1, max(0, int(round(p * (n - 1)))))
    return sorted_vals[idx]


@dataclass
class _Pending:
    send_ns: int
    payload_size: int


class StatsTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._rtts: deque[float] = deque(maxlen=RTT_WINDOW_SIZE)
        self._uplink_mbps: deque[float] = deque(maxlen=RTT_WINDOW_SIZE)
        self._downlink_mbps: deque[float] = deque(maxlen=RTT_WINDOW_SIZE)
        self.sent = 0
        self.received = 0
        self.stalls = 0
        self.drops = 0
        self.reconnects = 0
        self.last_rtt_ms: Optional[float] = None
        self.last_uplink_mbps: Optional[float] = None
        self.last_downlink_mbps: Optional[float] = None
        self.connection_state = "disconnected"
        self.endpoint = MODAL_QUIC_ENDPOINT
        self.last_error: Optional[str] = None
        self.server_info: dict = {}

    def mark_sent(self, seq: int, send_ns: int, payload_size: int) -> None:
        with self._lock:
            self.sent += 1
            self._pending[seq] = _Pending(send_ns=send_ns, payload_size=payload_size)

    def claim_pending(self, seq: int) -> Optional[_Pending]:
        with self._lock:
            return self._pending.pop(seq, None)

    def record_rtt(self, rtt_ms: float) -> None:
        with self._lock:
            self.received += 1
            self._rtts.append(rtt_ms)
            self.last_rtt_ms = rtt_ms

    def record_throughput(
        self, uplink_ms: float, downlink_ms: float, payload_size: int
    ) -> None:
        """Push per-request derived uplink/downlink Mbps into rolling windows.

        Ignores rows where the one-way estimate is <= 1 ms because at that
        point uncorrected clock skew dominates the number.
        """
        bits = payload_size * 8
        with self._lock:
            if uplink_ms > 1.0:
                mbps = bits / (uplink_ms / 1000.0) / 1_000_000
                if 0 < mbps < 10_000:
                    self._uplink_mbps.append(mbps)
                    self.last_uplink_mbps = mbps
            if downlink_ms > 1.0:
                mbps = bits / (downlink_ms / 1000.0) / 1_000_000
                if 0 < mbps < 10_000:
                    self._downlink_mbps.append(mbps)
                    self.last_downlink_mbps = mbps

    def sweep_stalls(self, now_ns: int, timeout_ns: int) -> list[tuple[int, _Pending]]:
        expired: list[tuple[int, _Pending]] = []
        with self._lock:
            for seq, pending in list(self._pending.items()):
                if now_ns - pending.send_ns > timeout_ns:
                    expired.append((seq, pending))
                    del self._pending[seq]
            self.stalls += len(expired)
        return expired

    def drop_all_pending(self) -> int:
        with self._lock:
            n = len(self._pending)
            self._pending.clear()
            return n

    def percentile(self, p: float) -> Optional[float]:
        with self._lock:
            samples = sorted(self._rtts)
        return _percentile(samples, p)

    def uplink_percentile(self, p: float) -> Optional[float]:
        with self._lock:
            samples = sorted(self._uplink_mbps)
        return _percentile(samples, p)

    def downlink_percentile(self, p: float) -> Optional[float]:
        with self._lock:
            samples = sorted(self._downlink_mbps)
        return _percentile(samples, p)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "sent": self.sent,
                "received": self.received,
                "stalls": self.stalls,
                "drops": self.drops,
                "reconnects": self.reconnects,
                "in_flight": len(self._pending),
                "last_rtt_ms": self.last_rtt_ms,
                "last_uplink_mbps": self.last_uplink_mbps,
                "last_downlink_mbps": self.last_downlink_mbps,
                "connection_state": self.connection_state,
                "endpoint": self.endpoint,
                "last_error": self.last_error,
                "window_size": len(self._rtts),
                "server_info": dict(self.server_info),
            }


# --------------------------------------------------------------------------- #
# Connection manager
# --------------------------------------------------------------------------- #


class ConnectionManager:
    """Owns the QUIC portal and the sender/receiver/sweeper threads."""

    def __init__(self, tracker: StatsTracker, recorder: Recorder) -> None:
        self.tracker = tracker
        self.recorder = recorder
        self._lock = threading.Lock()
        self._portal = None
        self._stop = threading.Event()
        self._reconnect_event = threading.Event()
        # Tuple swap is atomic under the GIL, so the sender can read it lock-free.
        self._payload_info: tuple[int, bytes] = (
            PAYLOAD_SIZE_BYTES,
            secrets.token_bytes(PAYLOAD_SIZE_BYTES),
        )
        self._next_seq = 1
        self._seq_lock = threading.Lock()
        self._send_lock = threading.Lock()  # portal.send is not thread-safe
        self._sender_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self._sweeper_thread: Optional[threading.Thread] = None
        self._connect_thread: Optional[threading.Thread] = None
        self._modal_app = None
        self._coord_dict = None
        self._server_fn = None
        self._server_call = None

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        self._sender_thread = threading.Thread(target=self._sender_loop, name="sender", daemon=True)
        self._receiver_thread = threading.Thread(target=self._receiver_loop, name="receiver", daemon=True)
        self._sweeper_thread = threading.Thread(target=self._sweeper_loop, name="sweeper", daemon=True)
        self._connect_thread = threading.Thread(target=self._connect_loop, name="connect", daemon=True)
        self._sender_thread.start()
        self._receiver_thread.start()
        self._sweeper_thread.start()
        self._connect_thread.start()

    @property
    def payload_size(self) -> int:
        return self._payload_info[0]

    def set_payload_size(self, size: int) -> None:
        if size < 1 or size > 10_000_000:
            raise ValueError("payload size must be in [1, 10_000_000] bytes")
        self._payload_info = (size, secrets.token_bytes(size))
        LOG.info("payload size set to %d B", size)

    def stop(self) -> None:
        self._stop.set()
        self._reconnect_event.set()
        with self._lock:
            portal = self._portal
            self._portal = None
        if portal is not None:
            try:
                portal.close()
            except Exception:
                pass
        if self._server_call is not None:
            try:
                self._server_call.cancel()
            except Exception:
                pass

    # -- connection ------------------------------------------------------- #

    def _set_state(self, state: str, error: Optional[str] = None) -> None:
        self.tracker.connection_state = state
        if error is not None:
            self.tracker.last_error = error
        LOG.info("connection state -> %s%s", state, f" ({error})" if error else "")

    def _connect_loop(self) -> None:
        """Establish (and re-establish) the QUIC connection."""
        import modal  # imported lazily so the script can still print --help without modal

        backoff = 1.0
        while not self._stop.is_set():
            self._reconnect_event.clear()
            self._set_state("connecting")
            try:
                if self._modal_app is None:
                    LOG.info("looking up modal app=%s function=%s", APP_NAME, SERVER_FUNCTION)
                    self._server_fn = modal.Function.from_name(APP_NAME, SERVER_FUNCTION)
                    self._coord_dict = modal.Dict.from_name(
                        COORD_DICT_NAME, create_if_missing=True
                    )
                    self._modal_app = True

                # Clear the dict so quic-portal sees a fresh rendezvous slot.
                try:
                    self._coord_dict.clear()
                except Exception as exc:  # pragma: no cover - best effort
                    LOG.warning("coord_dict.clear() failed: %s", exc)

                LOG.info("spawning server function on Modal...")
                self._server_call = self._server_fn.spawn()
                # Give the warm container a moment to enter create_server.
                time.sleep(2.0)

                LOG.info("creating local QUIC client portal on port %d", CLIENT_LOCAL_PORT)
                from quic_portal import Portal

                portal = Portal.create_client(
                    dict=self._coord_dict, local_port=CLIENT_LOCAL_PORT
                )
                with self._lock:
                    self._portal = portal
                try:
                    info = self._coord_dict.get("server_info", default=None)
                    if isinstance(info, dict):
                        self.tracker.server_info = info
                        LOG.info(
                            "server landed on cloud=%s region=%s (requested=%s) task=%s",
                            info.get("modal_cloud_provider"),
                            info.get("modal_region"),
                            info.get("requested_region"),
                            info.get("modal_task_id"),
                        )
                except Exception as exc:  # pragma: no cover - best effort
                    LOG.warning("could not read server_info: %s", exc)
                self._set_state("connected", error=None)
                backoff = 1.0

                # Block until somebody trips the reconnect event (recv error,
                # stall storm, or shutdown).
                self._reconnect_event.wait()
                if self._stop.is_set():
                    break

                self._set_state("reconnecting")
                self.tracker.reconnects += 1
                self.tracker.drops += 1
                self.tracker.drop_all_pending()
                self.recorder.write_event("drop")
                self._teardown_portal()
            except Exception as exc:
                LOG.exception("connect attempt failed")
                self._set_state("reconnecting", error=str(exc))
                self._teardown_portal()
                wait_s = min(30.0, backoff)
                if self._stop.wait(wait_s):
                    break
                backoff = min(30.0, backoff * 2)
                continue

            if self._stop.is_set():
                break

        self._set_state("disconnected")

    def _teardown_portal(self) -> None:
        with self._lock:
            portal = self._portal
            self._portal = None
        if portal is not None:
            try:
                portal.close()
            except Exception:
                pass
        if self._server_call is not None:
            try:
                self._server_call.cancel()
            except Exception:
                pass
            self._server_call = None

    def _trigger_reconnect(self, reason: str) -> None:
        if self.tracker.connection_state == "connected":
            LOG.warning("triggering reconnect: %s", reason)
            self.tracker.last_error = reason
            self._reconnect_event.set()

    def _get_portal(self):
        with self._lock:
            return self._portal

    # -- worker loops ----------------------------------------------------- #

    def _sender_loop(self) -> None:
        period = 1.0 / max(0.001, REQUEST_RATE_HZ)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now_mono = time.monotonic()
            if now_mono < next_tick:
                time.sleep(min(period, next_tick - now_mono))
                continue
            next_tick += period
            # If we fell badly behind, skip ahead instead of bursting.
            if time.monotonic() - next_tick > period * 5:
                next_tick = time.monotonic() + period

            portal = self._get_portal()
            if portal is None:
                continue

            with self._seq_lock:
                seq = self._next_seq
                self._next_seq += 1
            send_ns = time.time_ns()
            payload_size, payload = self._payload_info  # atomic snapshot
            header = CLIENT_HDR.pack(seq, send_ns, payload_size)
            packet = header + payload
            try:
                with self._send_lock:
                    portal.send(packet)
                self.tracker.mark_sent(seq, send_ns, payload_size)
            except Exception as exc:
                LOG.warning("send failed seq=%d: %s", seq, exc)
                self._trigger_reconnect(f"send error: {exc}")

    def _receiver_loop(self) -> None:
        client_hdr_size = CLIENT_HDR.size  # noqa: F841 - kept for clarity
        srv_hdr_size = SERVER_HDR.size
        while not self._stop.is_set():
            portal = self._get_portal()
            if portal is None:
                time.sleep(0.05)
                continue
            try:
                data = portal.recv(timeout_ms=500)
            except Exception as exc:
                LOG.warning("recv failed: %s", exc)
                self._trigger_reconnect(f"recv error: {exc}")
                time.sleep(0.1)
                continue
            if data is None:
                continue
            recv_ns = time.time_ns()
            if len(data) < srv_hdr_size:
                LOG.warning("short response len=%d", len(data))
                continue
            seq, client_send_ns, srv_recv_ns, srv_send_ns, payload_len = SERVER_HDR.unpack_from(data, 0)
            pending = self.tracker.claim_pending(seq)
            if pending is None:
                # Already counted as a stall, or unknown seq.
                continue
            rtt_ms = (recv_ns - pending.send_ns) / 1e6
            self.tracker.record_rtt(rtt_ms)
            server_proc_ms = (srv_send_ns - srv_recv_ns) / 1e6
            uplink_ms_est = (srv_recv_ns - pending.send_ns) / 1e6
            downlink_ms_est = (recv_ns - srv_send_ns) / 1e6
            self.tracker.record_throughput(
                uplink_ms_est, downlink_ms_est, pending.payload_size
            )
            self.recorder.write_response(
                seq=seq,
                t_send_ns=pending.send_ns,
                t_server_recv_ns=srv_recv_ns,
                t_server_send_ns=srv_send_ns,
                t_recv_ns=recv_ns,
                rtt_ms=rtt_ms,
                uplink_ms_est=uplink_ms_est,
                downlink_ms_est=downlink_ms_est,
                server_proc_ms=server_proc_ms,
                payload_size=pending.payload_size,
            )

    def _sweeper_loop(self) -> None:
        timeout_ns = TIMEOUT_MS * 1_000_000
        while not self._stop.is_set():
            time.sleep(0.25)
            now_ns = time.time_ns()
            expired = self.tracker.sweep_stalls(now_ns, timeout_ns)
            if not expired:
                continue
            for seq, pending in expired:
                self.recorder.write_event(
                    "timeout",
                    seq=seq,
                    t_send_ns=pending.send_ns,
                    payload_size=pending.payload_size,
                )
            # If too many stalls accumulate, force a reconnect.
            if len(expired) >= max(5, int(REQUEST_RATE_HZ * 2)):
                self._trigger_reconnect(f"{len(expired)} consecutive stalls")


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #


tracker = StatsTracker()
recorder = Recorder(RECORDING_DIR)
manager = ConnectionManager(tracker, recorder)

api = FastAPI(title="Latency Bench")

INDEX_PATH = Path(__file__).resolve().parent / "index.html"


class StartRequest(BaseModel):
    label: str = "run"


class PayloadSizeRequest(BaseModel):
    payload_size_bytes: int


@api.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_PATH)


@api.get("/api/status")
def status() -> JSONResponse:
    snap = tracker.snapshot()
    snap.update(
        {
            "p50_ms": tracker.percentile(0.50),
            "p99_ms": tracker.percentile(0.99),
            "uplink_mbps_p50": tracker.uplink_percentile(0.50),
            "uplink_mbps_p90": tracker.uplink_percentile(0.90),
            "downlink_mbps_p50": tracker.downlink_percentile(0.50),
            "downlink_mbps_p90": tracker.downlink_percentile(0.90),
            "recording": {
                "active": recorder.active,
                "label": recorder.label,
                "path": recorder.path,
            },
            "config": {
                "payload_size_bytes": manager.payload_size,
                "request_rate_hz": REQUEST_RATE_HZ,
                "timeout_ms": TIMEOUT_MS,
                "recording_dir": str(RECORDING_DIR),
            },
        }
    )
    return JSONResponse(snap)


@api.post("/api/start")
def start(req: StartRequest) -> JSONResponse:
    try:
        path = recorder.start(req.label, tracker)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return JSONResponse({"ok": True, "path": path, "label": recorder.label})


@api.post("/api/stop")
def stop() -> JSONResponse:
    summary = recorder.stop(tracker)
    if summary is None:
        raise HTTPException(status_code=409, detail="no recording active")
    return JSONResponse(
        {
            "ok": True,
            "summary": {
                "label": summary.label,
                "path": summary.path,
                "started_at": summary.started_at,
                "stopped_at": summary.stopped_at,
                "duration_s": summary.duration_s,
                "total_requests": summary.total_requests,
                "responses": summary.responses,
                "stalls": summary.stalls,
                "drops": summary.drops,
                "stall_rate": summary.stall_rate,
                "rtt_min_ms": summary.rtt_min_ms,
                "rtt_p50_ms": summary.rtt_p50_ms,
                "rtt_p90_ms": summary.rtt_p90_ms,
                "rtt_p99_ms": summary.rtt_p99_ms,
                "rtt_max_ms": summary.rtt_max_ms,
            },
        }
    )


@api.post("/api/payload_size")
def set_payload_size(req: PayloadSizeRequest) -> JSONResponse:
    try:
        manager.set_payload_size(req.payload_size_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"ok": True, "payload_size_bytes": manager.payload_size})


@api.get("/api/recordings")
def recordings() -> JSONResponse:
    items = []
    if RECORDING_DIR.exists():
        for p in sorted(RECORDING_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
            st = p.stat()
            items.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": st.st_size,
                    "modified": st.st_mtime,
                }
            )
    return JSONResponse({"recordings": items})


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        LOG.info("signal %d received, shutting down", signum)
        recorder.stop(tracker)
        manager.stop()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def main() -> None:
    _install_signal_handlers()
    LOG.info(
        "config endpoint=%s payload=%dB rate=%.2fHz timeout=%dms recording_dir=%s",
        MODAL_QUIC_ENDPOINT,
        PAYLOAD_SIZE_BYTES,
        REQUEST_RATE_HZ,
        TIMEOUT_MS,
        RECORDING_DIR,
    )
    manager.start()
    try:
        uvicorn.run(api, host=UI_HOST, port=UI_PORT, log_level="info", access_log=False)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
