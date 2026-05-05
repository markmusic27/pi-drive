/*
 * pi-drive — IMU I2C Scanner & Validator (Arduino Mega 2560)
 *
 * Walks the I2C bus once every SCAN_INTERVAL_MS and prints every 7-bit
 * address that ACKs. Recognizes the addresses common 9-DoF IMUs default
 * to so we can identify the chip before writing a chip-specific driver.
 *
 * Wiring (Mega 2560):
 *   IMU VCC_3V3 -> Mega 3.3V        (DO NOT use 5V — module is 3.3V only)
 *   IMU GND     -> Mega GND
 *   IMU SDA     -> Mega pin 20 (SDA)
 *   IMU SCL     -> Mega pin 21 (SCL)
 *
 * Output over USB serial @ 115200:
 *   INFO,<text>
 *   FOUND,0x<addr>,guess=<chip name>
 *   ERR,<text>
 */

#include <Wire.h>

const uint32_t I2C_FREQ_HZ        = 100000UL;  // start at standard speed
const unsigned long SCAN_INTERVAL_MS = 2000UL;

unsigned long last_scan_ms = 0;

// Map common I2C addresses to the chip family that uses them as default
// or alt. Several IMUs collide on the same address — we'll narrow down
// in the next step (WHO_AM_I read) once we see what's on the bus.
const __FlashStringHelper *describe(uint8_t addr) {
  switch (addr) {
    case 0x28: return F("BNO055 (default) / BNO08x (alt)");
    case 0x29: return F("BNO055 (ADDR=high) / BNO08x (alt)");
    case 0x4A: return F("BNO080/BNO085/BNO086 (default)");
    case 0x4B: return F("BNO080/BNO085/BNO086 (ADDR=high)");
    case 0x68: return F("MPU6050 / MPU9250 / ICM20948 (AD0=low)");
    case 0x69: return F("MPU6050 / MPU9250 / ICM20948 (AD0=high)");
    case 0x6A: return F("LSM9DS1 accel+gyro / LSM6DSx (alt)");
    case 0x6B: return F("LSM6DSx (default) / LSM9DS1 (alt)");
    case 0x1C: return F("LSM9DS1 mag / LIS3MDL");
    case 0x1E: return F("LSM9DS1 mag (alt) / HMC5883L");
    case 0x76: return F("BMP280 / BME280 (SDO=low)");
    case 0x77: return F("BMP280 / BME280 (SDO=high)");
    default:   return F("unknown");
  }
}

void scan() {
  Serial.println(F("INFO,scanning I2C bus 0x03..0x77..."));
  uint8_t found = 0;
  for (uint8_t addr = 0x03; addr <= 0x77; ++addr) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      Serial.print(F("FOUND,0x"));
      if (addr < 0x10) Serial.print('0');
      Serial.print(addr, HEX);
      Serial.print(F(",guess="));
      Serial.println(describe(addr));
      ++found;
    }
  }
  if (found == 0) {
    Serial.println(F("ERR,no I2C devices detected"));
    Serial.println(F("ERR,check: 3.3V on VCC, SDA<->20, SCL<->21, GND, breakout pull-ups present"));
  } else {
    Serial.print(F("INFO,scan complete: "));
    Serial.print(found);
    Serial.println(F(" device(s) found"));
  }
}

void setup() {
  Serial.begin(115200);

  Wire.begin();
  Wire.setClock(I2C_FREQ_HZ);

  Serial.println(F("INFO,pi-drive IMU validation — I2C scanner starting"));
  Serial.print  (F("INFO,I2C @ "));
  Serial.print  (I2C_FREQ_HZ);
  Serial.println(F(" Hz on Mega SDA=20, SCL=21"));
}

void loop() {
  unsigned long now = millis();
  if (now - last_scan_ms >= SCAN_INTERVAL_MS) {
    last_scan_ms = now;
    scan();
    Serial.println(F("---"));
  }
}
