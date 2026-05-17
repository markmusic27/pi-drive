"""Sweep payload sizes against a running client; plot RTT/uplink/downlink vs size.

Usage:
    # First, start the client in another shell:
    #   UI_PORT=5050 python -m client.main
    # then run the sweep:
    python sweep.py
    python sweep.py --host 127.0.0.1:5050 --duration 20 --settle 3

Outputs:
    recordings/<label>_<size>.csv   one per size (the client writes them)
    recordings/sweep_<ts>_rtt.png
    recordings/sweep_<ts>_uplink.png
    recordings/sweep_<ts>_downlink.png
    recordings/sweep_<ts>_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SIZES = [
    10_000,
    25_000,
    50_000,
    100_000,
    200_000,
    350_000,
    500_000,
    750_000,
    1_000_000,
]


def _post(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = p * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[int(rank)]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate(csv_path: Path) -> dict:
    rtts: list[float] = []
    ups: list[float] = []
    dns: list[float] = []
    timeouts = 0
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event"):
                if row["event"] == "timeout":
                    timeouts += 1
                continue
            try:
                rtts.append(float(row["rtt_ms"]))
                ups.append(float(row["uplink_ms_est"]))
                dns.append(float(row["downlink_ms_est"]))
            except (KeyError, ValueError):
                continue
    rs, us, ds = sorted(rtts), sorted(ups), sorted(dns)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {
        "n": len(rtts),
        "timeouts": timeouts,
        "rtt_mean": mean(rtts),
        "rtt_p50": _percentile(rs, 0.50),
        "rtt_p90": _percentile(rs, 0.90),
        "rtt_min": rs[0] if rs else None,
        "rtt_max": rs[-1] if rs else None,
        "up_mean": mean(ups),
        "up_p50": _percentile(us, 0.50),
        "up_p90": _percentile(us, 0.90),
        "dn_mean": mean(dns),
        "dn_p50": _percentile(ds, 0.50),
        "dn_p90": _percentile(ds, 0.90),
    }


def _render_plot(
    sizes_kb: list[float],
    median: list[float | None],
    mean: list[float | None],
    *,
    ylabel: str,
    title: str,
    color: str,
    out: Path,
    show: bool,
) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return

    valid = [(s, m, a) for s, m, a in zip(sizes_kb, median, mean) if m is not None]
    if not valid:
        print(f"no data for {title}", file=sys.stderr)
        return
    sx = [v[0] for v in valid]
    sy_med = [v[1] for v in valid]
    sy_mean = [v[2] if v[2] is not None else v[1] for v in valid]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(sx, sy_med, marker="o", color=color, linewidth=2.0, label="median (p50)")
    ax.plot(
        sx,
        sy_mean,
        marker="x",
        linestyle="--",
        color=color,
        linewidth=1.0,
        alpha=0.55,
        label="mean (for comparison)",
    )
    ax.set_xlabel("payload size (KB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:5050", help="client host:port")
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="comma-separated payload sizes in bytes",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="seconds recording per size")
    parser.add_argument("--settle", type=float, default=3.0, help="seconds to wait after size change")
    parser.add_argument("--label-prefix", default="sweep")
    parser.add_argument("--no-show", action="store_true", help="do not pop up plot windows")
    parser.add_argument(
        "--rec-dir",
        type=Path,
        default=Path("recordings"),
        help="where the client writes CSVs (must match RECORDING_DIR)",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    base = f"http://{args.host}"
    sweep_ts = int(time.time())

    # connectivity check
    try:
        status = _get(f"{base}/api/status")
    except urllib.error.URLError as exc:
        sys.exit(f"could not reach client at {base}: {exc}")
    if status["connection_state"] != "connected":
        sys.exit(f"client not connected to server: state={status['connection_state']}")
    print(f"client connected ({status['server_info'].get('modal_cloud_provider', '?')}/"
          f"{status['server_info'].get('modal_region', '?')})")
    print(f"sweep_ts={sweep_ts}  sizes={sizes}  duration={args.duration}s  settle={args.settle}s")

    # If a recording is already active (e.g. user clicked Start in UI), stop it.
    if status["recording"]["active"]:
        print("stopping existing recording first...")
        _post(f"{base}/api/stop")

    rows: list[dict] = []
    for i, size in enumerate(sizes, 1):
        print(f"[{i}/{len(sizes)}] payload={size:>8} B  ({size/1024:.1f} KB)")
        _post(f"{base}/api/payload_size", {"payload_size_bytes": size})
        time.sleep(args.settle)
        label = f"{args.label_prefix}_{sweep_ts}_{size:07d}b"
        try:
            _post(f"{base}/api/start", {"label": label})
        except urllib.error.HTTPError as exc:
            print(f"  start failed: {exc}; aborting", file=sys.stderr)
            return
        time.sleep(args.duration)
        try:
            stop_res = _post(f"{base}/api/stop")
        except urllib.error.HTTPError as exc:
            print(f"  stop failed: {exc}; aborting", file=sys.stderr)
            return
        summary = stop_res["summary"]
        path = Path(summary["path"])
        agg = aggregate(path)
        agg.update({"size": size, "path": str(path), "label": label})
        rows.append(agg)
        print(
            f"  n={agg['n']:>4}  timeouts={agg['timeouts']}  "
            f"rtt p50={agg['rtt_p50']!s:>8}  uplink p50={agg['up_p50']!s:>8}  "
            f"downlink p50={agg['dn_p50']!s:>8}"
        )

    # Persist combined summary CSV.
    out_csv = args.rec_dir / f"sweep_{sweep_ts}_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "size",
                "n",
                "timeouts",
                "rtt_mean",
                "rtt_p50",
                "rtt_p90",
                "rtt_min",
                "rtt_max",
                "up_mean",
                "up_p50",
                "up_p90",
                "dn_mean",
                "dn_p50",
                "dn_p90",
                "label",
                "path",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")

    sizes_kb = [r["size"] / 1024 for r in rows]
    show = not args.no_show
    _render_plot(
        sizes_kb,
        [r["rtt_p50"] for r in rows],
        [r["rtt_mean"] for r in rows],
        ylabel="RTT (ms)",
        title="RTT vs payload size",
        color="#3b82f6",
        out=args.rec_dir / f"sweep_{sweep_ts}_rtt.png",
        show=show,
    )
    _render_plot(
        sizes_kb,
        [r["up_p50"] for r in rows],
        [r["up_mean"] for r in rows],
        ylabel="uplink one-way est. (ms)",
        title="Uplink time vs payload size",
        color="#22c55e",
        out=args.rec_dir / f"sweep_{sweep_ts}_uplink.png",
        show=show,
    )
    _render_plot(
        sizes_kb,
        [r["dn_p50"] for r in rows],
        [r["dn_mean"] for r in rows],
        ylabel="downlink one-way est. (ms)",
        title="Downlink time vs payload size",
        color="#f97316",
        out=args.rec_dir / f"sweep_{sweep_ts}_downlink.png",
        show=show,
    )


if __name__ == "__main__":
    main()
