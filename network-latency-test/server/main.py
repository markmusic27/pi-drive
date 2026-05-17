"""Modal-hosted QUIC echo server for the cart latency benchmark.

Deploy with:
    modal deploy server/main.py

The deployed app exposes a long-running `run_server` function. The client
(running on the Jetson Thor cart) spawns this function via the Modal SDK,
which causes it to land on the warm container kept alive by `min_containers=1`.
The server then performs NAT-punching with the client through the shared
`latency-bench-coord` Modal Dict and starts echoing framed packets back.

Wire protocol (little-endian, single bidirectional QUIC stream, one message
per `send`/`recv`):

    Client -> Server: [u64 client_seq][u64 client_send_ns][u32 payload_len][payload]
    Server -> Client: [u64 client_seq][u64 client_send_ns]
                      [u64 server_recv_ns][u64 server_send_ns]
                      [u32 payload_len][payload]

`server_recv_ns` and `server_send_ns` are captured from the server's
`time.time_ns()` clock so the client can estimate one-way uplink/downlink
latency when the two clocks are NTP-synced.
"""

from __future__ import annotations

import os
import struct
import time

import modal

APP_NAME = os.environ.get("LATENCY_BENCH_APP_NAME", "latency-bench")
COORD_DICT_NAME = os.environ.get("LATENCY_BENCH_COORD_DICT", f"{APP_NAME}-coord")

SERVER_PORT = int(os.environ.get("SERVER_LOCAL_PORT", "5555"))
SERVER_REGION = os.environ.get("SERVER_REGION", "us-west")
SERVER_GPU = os.environ.get("SERVER_GPU", "L4")
SERVER_TIMEOUT_S = int(os.environ.get("SERVER_TIMEOUT_S", str(8 * 60 * 60)))
IDLE_TIMEOUT_MS = int(os.environ.get("SERVER_IDLE_TIMEOUT_MS", "30000"))

CLIENT_HDR = struct.Struct("<QQI")
SERVER_HDR = struct.Struct("<QQQQI")

app = modal.App(APP_NAME)

# quic-portal ships a manylinux_2_34 x86_64 wheel on PyPI, and debian_slim
# (bookworm, glibc 2.36) satisfies that constraint — no Rust toolchain needed.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "quic-portal==0.1.13"
)

coord_dict = modal.Dict.from_name(COORD_DICT_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=SERVER_GPU,
    region=SERVER_REGION,
    min_containers=1,
    max_containers=1,
    timeout=SERVER_TIMEOUT_S,
)
def run_server() -> None:
    """Long-running echo loop. The outer loop re-listens after a client drop."""
    from quic_portal import Portal

    actual_region = os.environ.get("MODAL_REGION", "?")
    cloud_raw = os.environ.get("MODAL_CLOUD_PROVIDER", "?")
    # Modal exports it as e.g. "CLOUD_PROVIDER_AWS" — strip the prefix.
    cloud = cloud_raw.removeprefix("CLOUD_PROVIDER_").lower() if cloud_raw != "?" else "?"
    task_id = os.environ.get("MODAL_TASK_ID", "?")
    print(
        f"[server] task={task_id} requested_region={SERVER_REGION} "
        f"actual_region={actual_region} cloud={cloud} gpu={SERVER_GPU} port={SERVER_PORT}",
        flush=True,
    )

    while True:
        print("[server] clearing coord dict and waiting for client...", flush=True)
        try:
            coord_dict.clear()
            # Publish where we landed so the client can show it in the UI.
            coord_dict["server_info"] = {
                "modal_region": actual_region,
                "modal_cloud_provider": cloud,
                "modal_task_id": task_id,
                "requested_region": SERVER_REGION,
                "gpu": SERVER_GPU,
            }
        except Exception as exc:
            print(f"[server] coord_dict.clear() failed: {exc}", flush=True)

        portal = None
        try:
            portal = Portal.create_server(dict=coord_dict, local_port=SERVER_PORT)
            print("[server] client connected, entering echo loop", flush=True)
            _echo_loop(portal)
        except Exception as exc:
            print(f"[server] connection error: {exc!r}", flush=True)
        finally:
            if portal is not None:
                try:
                    portal.close()
                except Exception:
                    pass
            print("[server] connection closed, will re-listen", flush=True)
            time.sleep(0.25)


def _echo_loop(portal) -> None:
    client_hdr_size = CLIENT_HDR.size
    pack_server = SERVER_HDR.pack
    while True:
        data = portal.recv(timeout_ms=IDLE_TIMEOUT_MS)
        recv_ns = time.time_ns()
        if data is None:
            # No traffic for IDLE_TIMEOUT_MS: assume the client is gone.
            print("[server] idle timeout, dropping connection", flush=True)
            return
        if len(data) < client_hdr_size:
            print(f"[server] short packet len={len(data)}", flush=True)
            continue
        seq, client_send_ns, payload_len = CLIENT_HDR.unpack_from(data, 0)
        payload = data[client_hdr_size:]
        send_ns = time.time_ns()
        header = pack_server(seq, client_send_ns, recv_ns, send_ns, payload_len)
        portal.send(header + payload)


@app.local_entrypoint()
def smoke() -> None:
    """`modal run server/main.py` — quick sanity check that the image builds."""
    print(f"Deploy with: modal deploy server/main.py  (app: {APP_NAME})")
    print(f"Coord dict: {COORD_DICT_NAME}")
