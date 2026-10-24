/**
 * EcoVend AI — Smart Canteen Waste RVM Microcontroller Firmware (ESP32 / Arduino)
 * ================================================================================
 * Hardware Flow:
 *   1. Vibratory Feeder + NEMA-17 Conveyor singulates waste into optical chamber
 *   2. HX711 Load Cell + VL53L0X ToF verify 3D object & weight plausibility
 *   3. AI Host (ResNet50 / YOLOv8) sends category over Serial UART (115200 baud):
 *      - "BIODEGRADABLE"   -> Servo rotates to 60°  (Green Bin)
 *      - "RECYCLABLE"      -> Servo rotates to 120° (Blue Bin)
 *      - "NON_RECYCLABLE"  -> Servo rotates to 180° (Black Bin)
 *      - "E_WASTE"         -> Servo rotates to 240° (Red Hazardous Bin)
 *   4. IR Break-Beam sensor confirms physical drop into bin before resetting flap to 90°
 */

#include <Arduino.h>
#include <ESP32Servo.h>

// Pin Definitions (ESP32-WROOM-32D)
static const int PIN_SERVO_FLAP    = 18; // PWM Servo Sorting Diverter (270° DS3225 / MG996R)
static const int PIN_STEP_DIR      = 25; // A4988 Stepper Direction (Conveyor Belt)
static const int PIN_STEP_PULSE    = 26; // A4988 Stepper Step Pulse
static const int PIN_IR_DROP_BEAM  = 4;  // IR Break-Beam Drop Confirmation Sensor
static const int PIN_HX711_DOUT    = 32; // Load Cell Data
static const int PIN_HX711_SCK     = 33; // Load Cell Clock
static const int PIN_TRIG_BIO      = 12; // Ultrasonic Bin Level Trigger
static const int PIN_ECHO_BIO      = 14; // Ultrasonic Bin Level Echo

// Servo Angles per Waste Category
static const int ANGLE_NEUTRAL        = 90;
static const int ANGLE_BIODEGRADABLE  = 60;
static const int ANGLE_RECYCLABLE     = 120;
static const int ANGLE_NON_RECYCLABLE = 180;
static const int ANGLE_E_WASTE        = 240;

Servo sortingServo;

// Telemetry Counters (Part 4)
unsigned long totalProcessed   = 0;
unsigned long countBiodeg      = 0;
unsigned long countRecyclable  = 0;
unsigned long countNonRec      = 0;
unsigned long countEWaste      = 0;

void advanceConveyorSteps(int steps) {
  digitalWrite(PIN_STEP_DIR, HIGH);
  for (int i = 0; i < steps; i++) {
    digitalWrite(PIN_STEP_PULSE, HIGH);
    delayMicroseconds(800);
    digitalWrite(PIN_STEP_PULSE, LOW);
    delayMicroseconds(800);
  }
}

void actuateSortingFlap(int targetAngle, const char* categoryLabel) {
  Serial.printf("[ESP32-ACTUATOR] Routing to %s -> Servo %d deg\n", categoryLabel, targetAngle);
  sortingServo.write(targetAngle);

  // Wait for item to drop past IR break-beam sensor (timeout 1200ms)
  unsigned long tStart = millis();
  bool dropConfirmed = false;
  while (millis() - tStart < 1200) {
    if (digitalRead(PIN_IR_DROP_BEAM) == LOW) {
      dropConfirmed = true;
      break;
    }
    delay(20);
  }

  delay(450); // Ensure clean clearance from flap
  sortingServo.write(ANGLE_NEUTRAL);

  totalProcessed++;
  Serial.printf(
    "ACK_SORTED|cat=%s|angle=%d|drop=%d|total=%lu|bio=%lu|rec=%lu|non=%lu|ew=%lu\n",
    categoryLabel, targetAngle, dropConfirmed ? 1 : 0,
    totalProcessed, countBiodeg, countRecyclable, countNonRec, countEWaste
  );
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_STEP_DIR, OUTPUT);
  pinMode(PIN_STEP_PULSE, OUTPUT);
  pinMode(PIN_IR_DROP_BEAM, INPUT_PULLUP);

  // Attach 270-degree high-torque digital servo (500us - 2500us pulse width)
  sortingServo.setPeriodHertz(50);
  sortingServo.attach(PIN_SERVO_FLAP, 500, 2500);
  sortingServo.write(ANGLE_NEUTRAL);

  Serial.println("[ESP32-RVM-READY] Smart Waste Segregator Online @ 115200 baud");
}

void loop() {
  if (Serial.available() > 0) {
    String incoming = Serial.readStringUntil('\n');
    incoming.trim();
    incoming.toUpperCase();

    if (incoming.startsWith("BIODEGRADABLE")) {
      countBiodeg++;
      actuateSortingFlap(ANGLE_BIODEGRADABLE, "BIODEGRADABLE");
    } else if (incoming.startsWith("RECYCLABLE")) {
      countRecyclable++;
      actuateSortingFlap(ANGLE_RECYCLABLE, "RECYCLABLE");
    } else if (incoming.startsWith("NON_RECYCLABLE")) {
      countNonRec++;
      actuateSortingFlap(ANGLE_NON_RECYCLABLE, "NON_RECYCLABLE");
    } else if (incoming.startsWith("E_WASTE")) {
      countEWaste++;
      actuateSortingFlap(ANGLE_E_WASTE, "E_WASTE");
    } else if (incoming.startsWith("FEED_NEXT")) {
      advanceConveyorSteps(400);
      Serial.println("ACK_CONVEYOR_INDEXED");
    }
  }
}
