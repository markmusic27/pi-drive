# pi-drive

Code for a version of [$\pi_{0.5}$](https://www.physicalintelligence.company/blog/pi05) that can drive.

This repo runs the self-driving golf cart we've built at Stanford — a Club Car Precedent retrofitted with drive-by-wire steering and pedals, surround cameras, GPS/IMU, and a Jetson AGX Thor as the main compute node — and serves as the deployment platform for our CS224R project on fine-tuning $\pi_{0.5}$ (Physical Intelligence's generalist VLA) into a driving policy via actor-critic RL with NVIDIA's Alpamayo-R1 as the critic.

Today the cart is teleoperated from a PS5 controller; the goal is a $\pi_{0.5}$-based policy in the driver's seat.

## What's in here

- `limits.py`, `sketches/common/cart_limits.h` — shared mechanical/software limits (Python + Arduino, kept in sync).
- `sketches/` — Arduino Mega firmware (pedal control with watchdog, GPS/IMU passthroughs, bench jog tests).
- `scripts/` — host-side entrypoints (PS5 drive, camera grid, sensor monitors, firmware uploader).
- `main.py` — ODrive S1 steering sweep.
- `docs/` — per-subsystem writeups; start with [`docs/architecture.md`](docs/architecture.md).

## Running

```bash
uv sync
uv run python scripts/ps5_drive.py
```

Teleop demo: <https://youtu.be/j9FjoRA4yb0>
