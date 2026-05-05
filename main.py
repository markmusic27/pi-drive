"""
pi-drive - ODrive S1 Steering Test Script
==========================================
Controls the M8325s motor via ODrive S1 over USB-C.
Continuous sweep: 0° → +90° → 0° → -90° → 0°.
Uses trapezoidal trajectory mode for smooth acceleration/deceleration.

Hardware Setup:
- ODrive S1 + M8325s-100KV motor
- HTD 5M belt drive, 3:1 ratio (20T motor → 60T column)
- USB-C connection from Jetson AGX Thor through USB isolator

Usage:
    uv run python main.py
"""

import odrive
from odrive.enums import AxisState, InputMode
import time
import sys

from limits import (
    STEERING_BELT_RATIO as BELT_RATIO,
    STEERING_MAX_DEG,
    STEERING_MIN_DEG,
    steering_deg_to_motor_turns as deg_to_motor_turns,
    motor_turns_to_steering_deg as motor_turns_to_deg,
)

# --- Configuration ---
# Sweep amplitude is clamped to the soft steering limits from limits.py.
MAX_ANGLE_DEG = min(STEERING_MAX_DEG, -STEERING_MIN_DEG)
NUM_SWEEPS = 5             # number of full left-right sweeps
POSITION_THRESHOLD = 0.002  # turns - how close is "arrived" (~0.24° at column with 3:1 belt)

# Trajectory planner limits (motor-side units) — these are the MAX for the final sweep
TRAP_VEL_MAX    = 8.0      # turns/s  — peak cruise speed (sweep 5)
TRAP_ACCEL_MAX  = 15.0     # turns/s² — acceleration (sweep 5)
TRAP_DECEL_MAX  = 15.0     # turns/s² — deceleration (sweep 5)

def wait_for_position(axis, target_pos: float, timeout: float = 10.0):
    """Wait until the motor reaches the target position or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        current = axis.pos_estimate
        error = abs(current - target_pos)
        if error < POSITION_THRESHOLD:
            return True
        time.sleep(0.05)
    print(f"  Warning: position not reached within {timeout}s (error: {error:.4f} turns)")
    return False

def main():
    # --- Connect ---
    print("Connecting to ODrive...")
    try:
        odrv0 = odrive.find_any(timeout=10)
    except Exception as e:
        print(f"Error: Could not find ODrive. Is it connected via USB? ({e})")
        sys.exit(1)

    print(f"Connected to ODrive {odrv0.serial_number}")
    print(f"  Bus voltage: {odrv0.vbus_voltage:.1f}V")
    print(f"  FW version:  {odrv0.fw_version_major}.{odrv0.fw_version_minor}.{odrv0.fw_version_revision}")

    # --- Check for errors ---
    if odrv0.axis0.active_errors != 0:
        print(f"  Active errors detected: {odrv0.axis0.active_errors}")
        print("  Clearing errors...")
        odrv0.clear_errors()
        time.sleep(0.5)

    # --- Enter closed-loop control ---
    print("Entering closed-loop position control...")
    odrv0.axis0.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.5)

    if odrv0.axis0.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"Error: Failed to enter closed-loop control. State: {odrv0.axis0.current_state}")
        print(f"  Disarm reason: {odrv0.axis0.disarm_reason}")
        sys.exit(1)

    print("Closed-loop control active.")

    # --- Configure trapezoidal trajectory mode ---
    print("Configuring trapezoidal trajectory mode...")
    odrv0.axis0.controller.config.input_mode = InputMode.TRAP_TRAJ
    print(f"  Speed ramps from {TRAP_VEL_MAX / NUM_SWEEPS:.1f} to {TRAP_VEL_MAX:.1f} turns/s over {NUM_SWEEPS} sweeps")

    # --- Record starting position ---
    start_pos = odrv0.axis0.pos_estimate
    print(f"\nStarting position: {start_pos:.4f} turns")
    print(f"Sweep: 0° → +{MAX_ANGLE_DEG}° → 0° → -{MAX_ANGLE_DEG}° → 0° (×{NUM_SWEEPS})")

    waypoints = [MAX_ANGLE_DEG, 0.0, -MAX_ANGLE_DEG, 0.0]

    # --- Execute steering sequence ---
    try:
        for sweep in range(1, NUM_SWEEPS + 1):
            frac = sweep / NUM_SWEEPS
            vel = TRAP_VEL_MAX * frac
            accel = TRAP_ACCEL_MAX * frac
            decel = TRAP_DECEL_MAX * frac
            odrv0.axis0.trap_traj.config.vel_limit = vel
            odrv0.axis0.trap_traj.config.accel_limit = accel
            odrv0.axis0.trap_traj.config.decel_limit = decel

            print(f"\n--- Sweep {sweep}/{NUM_SWEEPS} (vel={vel:.1f}, accel={accel:.1f}) ---")
            for deg in waypoints:
                target = start_pos + deg_to_motor_turns(deg)
                print(f"\n>>> Moving to {deg:+.0f}°...")
                odrv0.axis0.controller.input_pos = target
                if wait_for_position(odrv0.axis0, target):
                    actual_deg = motor_turns_to_deg(odrv0.axis0.pos_estimate - start_pos)
                    print(f"  Arrived at {actual_deg:+.1f}°")

        print("\nHolding at 0° for 10 seconds...")
        time.sleep(10)

        print("\nSteering test complete.")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Returning to start position...")
        odrv0.axis0.controller.input_pos = start_pos
        time.sleep(2)

    finally:
        print("Disarming motor...")
        odrv0.axis0.controller.config.input_mode = InputMode.PASSTHROUGH
        odrv0.axis0.requested_state = AxisState.IDLE
        print("Done.")

if __name__ == "__main__":
    main()
