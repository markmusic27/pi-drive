/*
 * pi-drive — Brake Actuator Jog Test (Arduino Mega 2560)
 *
 * Bare-minimum diagnostic. Drives the BRAKE BTS7960 H-bridge based on
 * single-character commands typed into the Arduino IDE Serial Monitor.
 * No pot feedback, no host protocol.
 *
 * Open Serial Monitor at 115200 baud. Line-ending setting doesn't matter.
 *
 *   F   brake FORWARD  (extend / pedal down — brake ENGAGED)
 *   B   brake BACKWARD (retract / pedal up — brake RELEASED)
 *   S   stop
 *
 * Lowercase f / b / s also work.
 */

const int BRAKE_R_EN  = 6;
const int BRAKE_R_PWM = 4;
const int BRAKE_L_EN  = 7;
const int BRAKE_L_PWM = 5;

const uint8_t PWM_SPEED = 80;  // 0..255

void driveStop() {
  analogWrite(BRAKE_R_PWM, 0);
  analogWrite(BRAKE_L_PWM, 0);
}

void driveForward() {  // extend (press pedal — brake engaged)
  analogWrite(BRAKE_R_PWM, 0);
  analogWrite(BRAKE_L_PWM, PWM_SPEED);
}

void driveBackward() {  // retract (release pedal — brake released)
  analogWrite(BRAKE_L_PWM, 0);
  analogWrite(BRAKE_R_PWM, PWM_SPEED);
}

void handleChar(char c) {
  switch (c) {
    case 'F': case 'f':
      Serial.println(F("STATE: FORWARD"));
      driveForward();
      break;
    case 'B': case 'b':
      Serial.println(F("STATE: BACKWARD"));
      driveBackward();
      break;
    case 'S': case 's':
      Serial.println(F("STATE: STOPPED"));
      driveStop();
      break;
    case '\r': case '\n': case ' ': case '\t':
      break;
    default:
      Serial.print(F("WARN,unknown command: '"));
      Serial.print(c);
      Serial.println(F("' — use F / B / S"));
      break;
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(BRAKE_R_EN,  OUTPUT);
  pinMode(BRAKE_R_PWM, OUTPUT);
  pinMode(BRAKE_L_EN,  OUTPUT);
  pinMode(BRAKE_L_PWM, OUTPUT);

  // BTS7960 only conducts when both EN lines are high.
  digitalWrite(BRAKE_R_EN, HIGH);
  digitalWrite(BRAKE_L_EN, HIGH);

  driveStop();

  Serial.println(F("INFO,brake actuator jog ready"));
  Serial.println(F("INFO,commands: F=forward(engage)  B=backward(release)  S=stop"));
  Serial.print  (F("INFO,PWM_SPEED=")); Serial.println(PWM_SPEED);
}

void loop() {
  while (Serial.available()) {
    handleChar((char)Serial.read());
  }
}
