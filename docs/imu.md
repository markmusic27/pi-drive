# pi-drive — IMU

The cart's inertial measurement unit is a 9-DoF sensor-fusion module on the Arduino Mega's I2C bus. Right now we're at the **bus-validation stage** — the chip ACKs at its expected I2C address but no orientation data is being read yet. A proper SH-2 driver is on the open-TODO list below.

See [`steering.md`](./steering.md) for the cart power map and [`gps.md`](./gps.md) for the GPS validation pattern this doc mirrors.

---

## Hardware

| Component   | Part                                                  | Notes                                       |
|-------------|-------------------------------------------------------|---------------------------------------------|
| IMU module  | Purple BNO08x breakout (silkscreen `VCC_3V3`)         | Bosch + CEVA / Hillcrest 9-DoF sensor fusion |
| Chip family | **BNO080 / BNO085 / BNO086**                          | Confirmed by I2C scan — see [validation](#how-to-validate-the-imu) |
| Protocol    | I2C (SH-2 framing on top — needs a real driver)       | NOT plain register reads like a BNO055      |
| I2C address | **0x4B** (ADDR pad pulled high on this breakout)      | Default for the family is `0x4A` / `0x4B`   |
| Host MCU    | Arduino Mega 2560                                     | Hardware I2C: SDA = pin 20, SCL = pin 21    |
| Power rail  | Mega 3.3V (NOT 5V — module is 3.3V only)              |                                             |

The BNO08x family is fundamentally different from the older BNO055 (which is plain I2C register reads). BNO08x uses the SH-2 protocol — packetized, requires either Adafruit's `Adafruit BNO08x` library or SparkFun's `SparkFun_BNO080_Arduino_Library` to read sensor reports.

---

## Wiring

> **Validated:** the breakout ACKs at `0x4B` every scan cycle on the Mega's I2C bus, so the four wires below are confirmed correct.

| Arduino Mega pin | IMU module pin | Direction       |
|------------------|----------------|-----------------|
| 3.3V             | `VCC_3V3`      | Power           |
| GND              | `GND`          | Ground          |
| Pin 20 (SDA)     | `SDA/MISO/TX`  | bidirectional   |
| Pin 21 (SCL)     | `SCL/SCK/RX`   | bidirectional   |

The breakout's pin labels (`SDA/MISO/TX`, etc.) reflect the chip's three protocol modes — I2C, SPI, UART — selected via the `PS0` / `PS1` jumpers on the back. Default state (both unjumpered) is **I2C**, which is what we use.

`ADDR`, `INT`, `RST`, `CS` are not connected — fine for I2C-mode validation. `RST` will eventually be wired to a GPIO so the host can reset the IMU on demand; `INT` will be wired so the chip can signal "new sensor report available" instead of being polled.

---

## Power & level shifting caveat

The BNO08x is a 3.3V part. The Mega's I2C lines are **5V**. This works in the short term because:

- I2C is open-drain. Neither side actively pulls high — the breakout's onboard pull-ups tie the bus to 3.3V, and the Mega only ever pulls down.
- The chip's I/O pads tolerate a brief overshoot; pulling SDA/SCL up to 3.3V (not 5V) is what actually happens in practice.

But for the production install on the cart, plan on either:

- A bidirectional level shifter (BSS138 or similar) on SDA/SCL, **or**
- Move the IMU off the Mega and onto the Jetson (or a 3.3V auxiliary microcontroller).

This is intentionally validation-only wiring — it's safe enough for bench testing, not for a permanent install.

---

## Validation sketch & script

The current firmware is a **pure I2C bus scanner**. It walks `0x03..0x77`, prints every address that ACKs, and pretty-prints common IMU chip families for any address it finds. It's intentionally library-free so we can validate the bus before pulling in the SH-2 driver.

| File                                              | Purpose                                                 |
|---------------------------------------------------|---------------------------------------------------------|
| `sketches/imu_validation/imu_validation.ino`      | Mega-side I2C scanner. Prints `FOUND,0x<addr>` lines.   |
| `scripts/upload.py`                               | Compiles and flashes the sketch — see [architecture.md § Firmware workflow](./architecture.md#firmware-workflow-no-arduino-ide). |

We do **not** have a host-side parser script for the IMU yet (analogous to `scripts/sensor_test.py` for GPS). For validation you read the Mega's USB serial output directly with `arduino-cli monitor`, which `scripts/upload.py --monitor` will open for you.

---

## How to validate the IMU

The whole flow runs from the project root, no Arduino IDE involved. `scripts/upload.py` is the script that compiles and flashes the `.ino` file — never edit and run `.ino` files directly in the Arduino IDE alongside this workflow, since the IDE's Serial Monitor will fight the upload script for the port.

1. **Connect** the IMU as in the [wiring table](#wiring). Double-check `VCC_3V3` is on the Mega's 3.3V rail, not 5V.
2. **Compile and flash** the validation sketch to the Mega:
   ```bash
   uv run python scripts/upload.py imu_validation
   ```
3. **Open a serial monitor** to read the scanner's output:
   ```bash
   uv run python scripts/upload.py imu_validation --monitor
   ```
   (Or, if you've already flashed: `arduino-cli monitor -p <PORT> -c baudrate=115200`.)

You should see, every ~2 seconds:

```
INFO,pi-drive IMU validation — I2C scanner starting
INFO,I2C @ 100000 Hz on Mega SDA=20, SCL=21
INFO,scanning I2C bus 0x03..0x77...
FOUND,0x4B,guess=BNO080/BNO085/BNO086 (ADDR=high)
INFO,scan complete: 1 device(s) found
---
```

What each line tells you:

| Line                                  | Means                                                              |
|---------------------------------------|--------------------------------------------------------------------|
| `INFO,…starting`                      | Mega is running the validation sketch (not some other firmware).   |
| `FOUND,0x4B,guess=BNO0…`              | Power, SDA, SCL, GND, and pull-ups are all correct. ✅              |
| `ERR,no I2C devices detected`         | Some wire is wrong. Check VCC, GND, SDA→20, SCL→21, in that order. |
| Multiple `FOUND` lines                | More than one I2C device on the bus — fine, but flag if unexpected.|

If you see `ERR,no I2C devices detected`, the typical culprits in order of likelihood are:

1. **VCC not connected** to 3.3V (check the LED on the breakout, if any).
2. **SDA/SCL swapped** between Mega pins 20 and 21.
3. **GND not connected** — the Mega and IMU need a common ground.
4. The breakout's PS0/PS1 jumpers got soldered into a non-I2C mode (UART or SPI).

---

## Common I2C addresses (reference)

The validation sketch will print a guess from this table. Two chips can share an address — narrowing further requires a chip-specific WHO_AM_I read, which is the next step.

| Address  | Likely chip                                     |
|----------|--------------------------------------------------|
| `0x28`   | BNO055 (default) / BNO08x (alt)                  |
| `0x29`   | BNO055 (ADDR=high) / BNO08x (alt)                |
| `0x4A`   | BNO080 / BNO085 / BNO086 (default)               |
| `0x4B`   | BNO080 / BNO085 / BNO086 (ADDR=high) ← **us**    |
| `0x68`   | MPU6050 / MPU9250 / ICM20948 (AD0=low)           |
| `0x69`   | MPU6050 / MPU9250 / ICM20948 (AD0=high)          |
| `0x6A`   | LSM9DS1 accel+gyro / LSM6DSx (alt)               |
| `0x6B`   | LSM6DSx (default) / LSM9DS1 (alt)                |
| `0x76`   | BMP280 / BME280 (SDO=low)                        |
| `0x77`   | BMP280 / BME280 (SDO=high)                       |

---

## Open TODOs

- [ ] Install the **Adafruit BNO08x** library (`arduino-cli lib install "Adafruit BNO08x"`) and write `sketches/imu_test/imu_test.ino` — read product ID (definitively identifies BNO080 vs 085 vs 086), enable the rotation-vector report, stream quaternions over USB.
- [ ] Add a host-side parser to `scripts/sensor_test.py` (already documented in [`architecture.md`](./architecture.md) as "GPS + (future) IMU monitor"). Should print rate + magnitude of accel/gyro/quaternion at a fixed cadence.
- [ ] Wire `INT` to a Mega GPIO so the IMU can signal "new sensor report available" instead of being polled.
- [ ] Wire `RST` so the host can hard-reset the IMU on demand (e.g. on watchdog timeout).
- [ ] Add a bidirectional level shifter (BSS138) on SDA/SCL, **or** move the IMU off the Mega entirely. The current 5V-master / 3.3V-slave wiring is validation-only.
- [ ] Decide where the IMU lives in the production stack — Mega or directly on the Jetson — and document the chosen plan here.
- [ ] Define the structured IMU message the Mega sends to the Jetson (frame rate, units, frame convention).
