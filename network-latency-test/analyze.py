"""Summary stats + RTT-over-time plot for a single recording CSV.

Usage:
    python analyze.py recordings/drive_tree_cover_run1_1715900000.csv
    python analyze.py recordings/foo.csv --out recordings/foo.png --no-show

The CSV is the format written by `client/main.py`. Anomaly rows (where the
`event` column is set) are excluded from RTT stats but counted separately.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = p * (len(sorted_values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[int(rank)]
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def load(csv_path: Path):
    samples: list[dict] = []
    events: list[dict] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event"):
                events.append(row)
                continue
            try:
                samples.append(
                    {
                        "seq": int(row["seq"]),
                        "t_send_ns": int(row["t_send_ns"]),
                        "t_recv_ns": int(row["t_recv_ns"]),
                        "rtt_ms": float(row["rtt_ms"]),
                        "uplink_ms_est": float(row["uplink_ms_est"]),
                        "downlink_ms_est": float(row["downlink_ms_est"]),
                        "server_proc_ms": float(row["server_proc_ms"]),
                        "payload_size": int(row["payload_size"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return samples, events


def summarize(samples: list[dict], events: list[dict]) -> dict:
    rtts = sorted(s["rtt_ms"] for s in samples)
    uplinks = sorted(s["uplink_ms_est"] for s in samples)
    downlinks = sorted(s["downlink_ms_est"] for s in samples)
    server_proc = sorted(s["server_proc_ms"] for s in samples)
    if samples:
        t0 = min(s["t_send_ns"] for s in samples)
        t1 = max(s["t_recv_ns"] for s in samples)
        duration_s = (t1 - t0) / 1e9
    else:
        duration_s = 0.0

    counts = Counter(e["event"] for e in events)
    uplink_mbps, downlink_mbps = derive_throughputs(samples)
    return {
        "n": len(samples),
        "duration_s": duration_s,
        "rtt_min": rtts[0] if rtts else None,
        "rtt_p50": _percentile(rtts, 0.50),
        "rtt_p90": _percentile(rtts, 0.90),
        "rtt_p99": _percentile(rtts, 0.99),
        "rtt_max": rtts[-1] if rtts else None,
        "rtt_mean": sum(rtts) / len(rtts) if rtts else None,
        "uplink_p50": _percentile(uplinks, 0.50),
        "downlink_p50": _percentile(downlinks, 0.50),
        "server_proc_p50": _percentile(server_proc, 0.50),
        "uplink_mbps_p50": _percentile(sorted(uplink_mbps), 0.50) if uplink_mbps else None,
        "uplink_mbps_p90": _percentile(sorted(uplink_mbps), 0.90) if uplink_mbps else None,
        "downlink_mbps_p50": _percentile(sorted(downlink_mbps), 0.50) if downlink_mbps else None,
        "downlink_mbps_p90": _percentile(sorted(downlink_mbps), 0.90) if downlink_mbps else None,
        "events": dict(counts),
    }


def _throughput_mbps(payload_bytes: int, transfer_ms: float) -> float | None:
    """Effective Mbps for moving payload_bytes in transfer_ms (one direction)."""
    if transfer_ms <= 0:
        return None
    bits = payload_bytes * 8
    seconds = transfer_ms / 1000.0
    return bits / seconds / 1_000_000


def derive_throughputs(samples: list[dict]) -> tuple[list[float], list[float]]:
    """Per-sample uplink/downlink Mbps from one-way time estimates."""
    uplink: list[float] = []
    downlink: list[float] = []
    for s in samples:
        # Ignore tiny/negative estimates (clock skew or bad rows).
        if s["uplink_ms_est"] > 1.0:
            mbps = _throughput_mbps(s["payload_size"], s["uplink_ms_est"])
            if mbps is not None and 0 < mbps < 2000:
                uplink.append(mbps)
        if s["downlink_ms_est"] > 1.0:
            mbps = _throughput_mbps(s["payload_size"], s["downlink_ms_est"])
            if mbps is not None and 0 < mbps < 2000:
                downlink.append(mbps)
    return uplink, downlink


def throughput_time_series(
    samples: list[dict],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """(uplink_t, uplink_mbps, downlink_t, downlink_mbps) aligned to send time."""
    if not samples:
        return [], [], [], []
    t0 = min(s["t_send_ns"] for s in samples)
    uplink_t: list[float] = []
    uplink_mbps: list[float] = []
    downlink_t: list[float] = []
    downlink_mbps: list[float] = []
    for s in samples:
        t = (s["t_send_ns"] - t0) / 1e9
        if s["uplink_ms_est"] > 1.0:
            mbps = _throughput_mbps(s["payload_size"], s["uplink_ms_est"])
            if mbps is not None and 0 < mbps < 2000:
                uplink_t.append(t)
                uplink_mbps.append(mbps)
        if s["downlink_ms_est"] > 1.0:
            mbps = _throughput_mbps(s["payload_size"], s["downlink_ms_est"])
            if mbps is not None and 0 < mbps < 2000:
                downlink_t.append(t)
                downlink_mbps.append(mbps)
    return uplink_t, uplink_mbps, downlink_t, downlink_mbps


def _fmt(v):
    return "—" if v is None else f"{v:.2f}"


def print_summary(path: Path, summary: dict) -> None:
    print(f"== {path} ==")
    print(f"samples:           {summary['n']}")
    print(f"duration:          {summary['duration_s']:.2f} s")
    print(f"rtt min/p50/p90/p99/max (ms): "
          f"{_fmt(summary['rtt_min'])} / {_fmt(summary['rtt_p50'])} / "
          f"{_fmt(summary['rtt_p90'])} / {_fmt(summary['rtt_p99'])} / {_fmt(summary['rtt_max'])}")
    print(f"rtt mean (ms):     {_fmt(summary['rtt_mean'])}")
    print(f"uplink_est p50:    {_fmt(summary['uplink_p50'])} ms")
    print(f"downlink_est p50:  {_fmt(summary['downlink_p50'])} ms")
    print(f"server_proc p50:   {_fmt(summary['server_proc_p50'])} ms")
    if summary.get("uplink_mbps_p50") is not None:
        print(
            f"uplink Mbps p50/p90:   {_fmt(summary['uplink_mbps_p50'])} / "
            f"{_fmt(summary['uplink_mbps_p90'])}"
        )
        print(
            f"downlink Mbps p50/p90: {_fmt(summary['downlink_mbps_p50'])} / "
            f"{_fmt(summary['downlink_mbps_p90'])}"
        )
    if summary["events"]:
        print("events:")
        for k, v in summary["events"].items():
            print(f"  {k}: {v}")
    else:
        print("events:            none")


def plot(samples: list[dict], events: list[dict], out: Path, show: bool) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed (`pip install matplotlib`); skipping plot.",
              file=sys.stderr)
        return

    if not samples:
        print("no samples; skipping plot", file=sys.stderr)
        return

    t0 = min(s["t_send_ns"] for s in samples)
    ts = [(s["t_send_ns"] - t0) / 1e9 for s in samples]
    rtts = [s["rtt_ms"] for s in samples]
    rtt_mean = sum(rtts) / len(rtts)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ts, rtts, linewidth=0.7, color="#3b82f6", label="rtt (ms)")
    ax.axhline(
        rtt_mean,
        color="#f97316",
        linestyle="--",
        linewidth=1.2,
        label=f"mean ({rtt_mean:.1f} ms)",
    )
    ax.set_xlabel("seconds since first send")
    ax.set_ylabel("rtt (ms)")
    ax.set_title(f"RTT over time ({len(samples)} samples)")
    ax.grid(alpha=0.3)

    for e in events:
        try:
            t_ev = (int(e["t_recv_ns"]) - t0) / 1e9
        except (KeyError, ValueError):
            continue
        color = {"timeout": "#fbbf24", "drop": "#ef4444", "reconnect": "#a855f7"}.get(
            e.get("event", ""), "#9ca3af"
        )
        ax.axvline(t_ev, color=color, alpha=0.5, linewidth=0.8)

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote plot {out}")
    if show:
        plt.show()


def plot_throughput_histogram(samples: list[dict], out: Path, show: bool) -> None:
    """Histogram of per-request effective uplink/downlink Mbps (derived from CSV)."""
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping throughput histogram.", file=sys.stderr)
        return

    uplink, downlink = derive_throughputs(samples)
    if not uplink and not downlink:
        print("no throughput samples; skipping histogram", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    def _hist(ax, values: list[float], title: str, color: str) -> None:
        if not values:
            ax.set_visible(False)
            return
        mean = sum(values) / len(values)
        ax.hist(values, bins=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(mean, color="#f97316", linestyle="--", linewidth=1.2, label=f"mean {mean:.1f}")
        ax.set_xlabel("Mbps")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25, axis="y")

    _hist(axes[0], uplink, f"Uplink throughput ({len(uplink)} samples)", "#3b82f6")
    _hist(axes[1], downlink, f"Downlink throughput ({len(downlink)} samples)", "#22c55e")

    fig.suptitle("Effective WiFi/link speed per request (from payload / one-way time est.)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote plot {out}")
    if show:
        plt.show()


def plot_throughput_over_time(
    samples: list[dict], events: list[dict], out: Path, show: bool
) -> None:
    """Uplink/downlink Mbps vs seconds since first send."""
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping throughput time plot.", file=sys.stderr)
        return

    upl_t, upl_m, dn_t, dn_m = throughput_time_series(samples)
    if not upl_m and not dn_m:
        print("no throughput samples; skipping Mbps vs time plot", file=sys.stderr)
        return

    t0 = min(s["t_send_ns"] for s in samples)
    fig, ax = plt.subplots(figsize=(11, 4.5))

    if upl_m:
        upl_mean = sum(upl_m) / len(upl_m)
        ax.plot(upl_t, upl_m, linewidth=0.7, color="#3b82f6", label="uplink Mbps")
        ax.axhline(
            upl_mean,
            color="#3b82f6",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"uplink mean ({upl_mean:.1f})",
        )
    if dn_m:
        dn_mean = sum(dn_m) / len(dn_m)
        ax.plot(dn_t, dn_m, linewidth=0.7, color="#22c55e", label="downlink Mbps")
        ax.axhline(
            dn_mean,
            color="#22c55e",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"downlink mean ({dn_mean:.1f})",
        )

    for e in events:
        try:
            t_ev = (int(e["t_recv_ns"]) - t0) / 1e9
        except (KeyError, ValueError):
            continue
        color = {"timeout": "#fbbf24", "drop": "#ef4444", "reconnect": "#a855f7"}.get(
            e.get("event", ""), "#9ca3af"
        )
        ax.axvline(t_ev, color=color, alpha=0.5, linewidth=0.8)

    ax.set_xlabel("seconds since first send")
    ax.set_ylabel("Mbps")
    ax.set_title(f"Effective throughput over time ({len(samples)} samples)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote plot {out}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="recording CSV to analyse")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    parser.add_argument("--no-show", action="store_true", help="do not pop up the plot window")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"file not found: {args.csv}")

    samples, events = load(args.csv)
    summary = summarize(samples, events)
    print_summary(args.csv, summary)

    show = not args.no_show
    out = args.out or args.csv.with_suffix(".png")
    plot(samples, events, out, show=show)

    hist_out = args.csv.with_name(args.csv.stem + "_speed_hist.png")
    plot_throughput_histogram(samples, hist_out, show=show)

    time_out = args.csv.with_name(args.csv.stem + "_speed_time.png")
    plot_throughput_over_time(samples, events, time_out, show=show)


if __name__ == "__main__":
    main()
