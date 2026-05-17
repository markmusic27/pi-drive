# Latency Benchmark Tool

End-to-end RTT measurement between a cart-mounted client (Jetson Thor +
Starlink Mini) and a Modal-hosted GPU container, over a single bidirectional
QUIC stream provided by [`quic-portal`](https://github.com/gongy/quic-portal).

Operator presses **Start Recording** before driving, **Stop Recording** when
done. Per-request rows are written to CSV in `./recordings/` for offline
analysis with `analyze.py`.

```
network-latency-test/
├── server/main.py     # Modal app (echo server)
├── client/main.py     # cart-side script (QUIC client + FastAPI UI)
├── client/index.html  # local UI served on http://localhost:5000
├── analyze.py         # CSV → summary stats + RTT plot
├── recordings/        # CSVs land here
└── pyproject.toml
```

## 1. Prerequisites

- Python **3.11+** on both the dev box (for `modal deploy`) and the cart.
- A [Modal](https://modal.com) account with `modal token new` configured.
- On the cart (Jetson Thor, aarch64): a working Rust toolchain in case
  pre-built `quic-portal` wheels aren't available for your platform — see
  *Installing `quic-portal` on Jetson Thor* below.
- The cart's clock should be NTP-synced with `chronyd` so the
  `uplink_ms_est` / `downlink_ms_est` columns are meaningful.

## 2. Deploy the server

From a dev machine with `modal` configured:

```bash
cd network-latency-test
pip install modal
modal deploy server/main.py
```

This deploys a Modal app named **`latency-bench`** with:

- region pinned to `us-west` (override with `SERVER_REGION` env var)
- GPU `L4` (cheapest, override with `SERVER_GPU`)
- `min_containers=1`, `max_containers=1` → one warm container kept alive
- a named `modal.Dict` called `latency-bench-coord` used as the
  quic-portal rendezvous

The server is a pure echo: it does not run any model. The GPU is requested
purely to satisfy the spec (compute is irrelevant).

> **Note on the `MODAL_QUIC_ENDPOINT` env var.** `quic-portal` does the
> NAT-punch through Modal Dict, so there's no static `host:port` to point at.
> The client picks up the endpoint as `modal://{app}/{function}` which is
> stored in `MODAL_QUIC_ENDPOINT` for display purposes. If you renamed the
> app or function, set this var on the client.

## 3. Install the client on the cart

```bash
git clone <this repo> /opt/cart
cd /opt/cart/network-latency-test
# Any Python 3.11+ works. Use `python3 --version` to check.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> **Heads-up on shadowed Pythons.** If you have miniconda/anaconda installed,
> bare `python` may resolve to conda's interpreter even when `pip3` and
> `python3` point to a system framework Python. Always activate the venv
> (`source .venv/bin/activate`) before running anything in this folder, or you
> will hit `ModuleNotFoundError` for packages that *are* installed — just not
> in the Python `python` is pointing to. Run `which python` to confirm.

Configure `modal` auth on the cart (one-time):

```bash
modal token new
```

### Installing `quic-portal` on Jetson Thor (aarch64)

PyPI only ships wheels for a few platforms. On Jetson Thor you will probably
need to build from source. Install once:

```bash
sudo apt-get install -y build-essential pkg-config libssl-dev curl git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install maturin
git clone https://github.com/gongy/quic-portal /tmp/quic-portal
cd /tmp/quic-portal && maturin build --release && pip install target/wheels/*.whl
```

If the install completes and `python -c "from quic_portal import Portal"`
succeeds, you're good.

## 4. Run the client

```bash
cd network-latency-test
python -m client.main
# or: python client/main.py
```

You should see logs ending in:

```
INFO uvicorn.main | Uvicorn running on http://127.0.0.1:5000
```

Open `http://localhost:5000` in a browser on the Thor (or, with `UI_HOST=0.0.0.0`,
from any device on the same network). The UI shows:

- **Connection** state: `connecting → connected`. If it stays in
  `reconnecting`, check the error banner.
- **Live RTT**: last, p50, p99 over a rolling 1024-sample window.
- **Counters**: sent / received / in flight / stalls / drops.
- **Start Recording**: type a label (e.g. `drive_tree_cover_run1`), hit the
  big green button. The label is sanitised and prefixed onto a file like
  `recordings/drive_tree_cover_run1_1715900000.csv`.

The client is always sending in the background at `REQUEST_RATE_HZ` — the
button only controls whether the rows hit disk. Stopping shows a summary
panel and the new file appears in *Past recordings*.

### Configuration

Set as env vars before starting the client. Defaults shown.

| Variable                 | Default                                     | Meaning |
| ------------------------ | ------------------------------------------- | ------- |
| `MODAL_APP_NAME`         | `latency-bench`                             | Modal app name to look up |
| `MODAL_SERVER_FUNCTION`  | `run_server`                                | Modal function name |
| `MODAL_COORD_DICT`       | `latency-bench-coord`                       | Modal Dict for quic-portal rendezvous |
| `MODAL_QUIC_ENDPOINT`    | `modal://latency-bench/run_server`          | Display string only |
| `PAYLOAD_SIZE_BYTES`     | `500000`                                    | Per-request payload size |
| `REQUEST_RATE_HZ`        | `5`                                         | Send rate |
| `TIMEOUT_MS`             | `2000`                                      | A response taking longer than this is a *stall* |
| `RECORDING_DIR`          | `./recordings/`                             | Where CSVs land |
| `CLIENT_LOCAL_PORT`      | `5556`                                      | Local QUIC port |
| `UI_HOST` / `UI_PORT`    | `127.0.0.1` / `5000`                        | FastAPI bind address |
| `FLUSH_EVERY_N`          | `50`                                        | Flush CSV every N rows |

## 5. NTP / `chronyd`

`uplink_ms_est` and `downlink_ms_est` are computed by subtracting the
server's `time.time_ns()` from the client's. They are only meaningful when
both clocks are NTP-synced. On the Thor:

```bash
sudo apt-get install -y chrony
sudo systemctl enable --now chronyd
chronyc tracking            # check 'System time' offset
chronyc sources             # check stratum / reach of selected peers
```

The Modal worker hosts run NTP by default, but their clock skew vs. the
cart's is the dominant error term. Treat the uplink/downlink columns as
**~±10 ms accurate at best**; only `rtt_ms` (a single-clock measurement) is
trustworthy at sub-millisecond resolution.

## 6. CSV format

One row per response (or anomaly event). Columns:

| Column              | Type   | Notes |
| ------------------- | ------ | ----- |
| `seq`               | u64    | Monotonic per-session sequence number |
| `t_send_ns`         | i64    | Client `time.time_ns()` when the request was handed to QUIC |
| `t_server_recv_ns`  | i64    | Server `time.time_ns()` when the request was received |
| `t_server_send_ns`  | i64    | Server `time.time_ns()` when the echo was sent |
| `t_recv_ns`         | i64    | Client `time.time_ns()` when the echo was received |
| `rtt_ms`            | f64    | `(t_recv_ns − t_send_ns) / 1e6` |
| `uplink_ms_est`     | f64    | `(t_server_recv_ns − t_send_ns) / 1e6` (needs synced clocks) |
| `downlink_ms_est`   | f64    | `(t_recv_ns − t_server_send_ns) / 1e6` (needs synced clocks) |
| `server_proc_ms`    | f64    | `(t_server_send_ns − t_server_recv_ns) / 1e6` |
| `payload_size`      | i32    | Payload bytes (excluding header) |
| `event`             | string | Empty for normal rows; `"timeout"`, `"drop"`, or `"reconnect"` for anomalies |

The file is flushed every `FLUSH_EVERY_N` rows (default 50) and on every
anomaly row, so a hard kill loses at most ~50 normal rows. On `SIGINT`/`SIGTERM`
the recorder closes cleanly.

## 7. Analyse

```bash
python analyze.py recordings/drive_tree_cover_run1_1715900000.csv
```

This prints summary stats and writes a `.png` alongside the CSV with RTT
over time, plus colored vertical lines for `timeout` (yellow), `drop` (red),
and `reconnect` (purple) events. Add `--no-show` for headless runs and
`--out path/to/plot.png` to override the output path.

`analyze.py` requires `numpy` and `matplotlib` — install with:

```bash
pip install -e '.[analyze]'
```

## 8. Out of scope

- No auth on the QUIC link — Modal's NAT-punched UDP is the only barrier.
- No real-time graphs in the UI beyond live counters.
- No GPS / Starlink obstruction correlation — operator names the run.
- No model inference on the server; it is a pure echo.
