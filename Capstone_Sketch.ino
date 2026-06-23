/*
  ===================================================================
  Smart Food Preservation System - ESP32 Control Code
  ===================================================================
  Hardware:
    - ESP32 ESP-WROOM-32
    - DHT22 (Temperature & Humidity)
    - TEC1-12706 Peltier Module (via MOSFET trigger switch)
    - Fan + Heat Sink (via same/second MOSFET channel)
    - KY-016 RGB LED (status indicator)
    - KY-012 Active Buzzer (alarm)
    - 0-25V Voltage Sensor Module (battery monitoring)

  Function:
    - Reads ambient temp/humidity from DHT22
    - Receives target temp/humidity from Blynk app (set by AI model
      after food identification, via virtual pins V2/V3)
    - Drives Peltier + Fan to maintain target conditions
    - Updates RGB LED color based on system status
    - Triggers buzzer if temp/humidity drifts outside safe range
    - Reports live data back to Blynk dashboard
    - Monitors battery voltage and warns on low battery
  ===================================================================
*/

#define BLYNK_TEMPLATE_ID "TMPL6EYrBb1Kv"
#define BLYNK_TEMPLATE_NAME "Freshguard AI"
#define BLYNK_AUTH_TOKEN "Hef6kq-IukqIiRkCpR-AHZ5ImXOVsAVW"

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <DHT.h>

// ---------------------- Pin Definitions ----------------------
#define DHTPIN 4  // DHT22 data pin
#define DHTTYPE DHT22

#define PELTIER_PIN 25  // MOSFET trigger -> Peltier module
#define FAN_PIN 26      // MOSFET trigger -> Fan (heat dissipation)

#define LED_RED_PIN 18  // KY-016 RGB LED
#define LED_GREEN_PIN 19
#define LED_BLUE_PIN 21

#define BUZZER_PIN 27  // KY-012 active buzzer

#define VOLT_SENSOR_PIN 34  // Analog pin for 0-25V voltage sensor

// ---------------------- Blynk Virtual Pins ----------------------
#define VPIN_TEMP V0          // Current temperature (output to app)
#define VPIN_HUMIDITY V1      // Current humidity (output to app)
#define VPIN_TARGET_TEMP V2   // Target temp from AI/food DB (input from app)
#define VPIN_TARGET_HUM V3    // Target humidity from AI/food DB (input from app)
#define VPIN_FOOD_LABEL V4    // Identified food name (input from app, display only)
#define VPIN_ALARM V5         // Alarm status (output to app)
#define VPIN_BATTERY V6       // Battery voltage (output to app)
#define VPIN_SYSTEM_STATE V7  // Text status e.g. "Cooling", "Idle", "Alarm"

// ---------------------- Wifi Credentials ----------------------
char ssid[] = "YourWiFiSSID";
char pass[] = "YourWiFiPassword";

// ---------------------- Globals ----------------------
DHT dht(DHTPIN, DHTTYPE);
BlynkTimer timer;

float currentTemp = 0.0;
float currentHumidity = 0.0;
float targetTemp = 4.0;       // Default safe fallback (typical fridge temp)
float targetHumidity = 85.0;  // Default fallback

const float TEMP_TOLERANCE = 1.5;     // +/- degrees C before triggering correction
const float TEMP_ALARM_MARGIN = 4.0;  // degrees C beyond target before alarm
const float HUM_ALARM_MARGIN = 15.0;  // % RH beyond target before alarm

bool alarmActive = false;

// ---------------------- Function: Read Battery Voltage ----------------------
float readBatteryVoltage() {
  int raw = analogRead(VOLT_SENSOR_PIN);
  // 0-25V sensor module typically outputs scaled analog signal.
  // Calibrate this factor against a multimeter reading for your specific module.
  float voltage = (raw / 4095.0) * 25.0;
  return voltage;
}

// ---------------------- Function: Set RGB LED Color ----------------------
void setStatusColor(int r, int g, int b) {
  analogWrite(LED_RED_PIN, r);
  analogWrite(LED_GREEN_PIN, g);
  analogWrite(LED_BLUE_PIN, b);
}

// Status color codes:
//   Green  -> within target range (stable)
//   Blue   -> actively cooling
//   Yellow -> approaching alarm threshold
//   Red    -> alarm condition

// ---------------------- Function: Sound Buzzer ----------------------
void soundAlarm(bool state) {
  digitalWrite(BUZZER_PIN, state ? HIGH : LOW);
}

// ---------------------- Blynk: Receive Target Temp from App ----------------------
BLYNK_WRITE(VPIN_TARGET_TEMP) {
  targetTemp = param.asFloat();
}

// ---------------------- Blynk: Receive Target Humidity from App ----------------------
BLYNK_WRITE(VPIN_TARGET_HUM) {
  targetHumidity = param.asFloat();
}

// ---------------------- Core Sensing + Control Logic ----------------------
void controlLoop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    Serial.println("DHT22 read failed, skipping this cycle.");
    return;
  }

  currentTemp = t;
  currentHumidity = h;

  // Push live sensor data to Blynk dashboard
  Blynk.virtualWrite(VPIN_TEMP, currentTemp);
  Blynk.virtualWrite(VPIN_HUMIDITY, currentHumidity);

  // --- Cooling control ---
  if (currentTemp > targetTemp + TEMP_TOLERANCE) {
    digitalWrite(PELTIER_PIN, HIGH);  // Activate Peltier cooling
    digitalWrite(FAN_PIN, HIGH);      // Activate fan for heat dissipation
    Blynk.virtualWrite(VPIN_SYSTEM_STATE, "Cooling");
    setStatusColor(0, 0, 255);  // Blue = cooling
  } else if (currentTemp < targetTemp - TEMP_TOLERANCE) {
    digitalWrite(PELTIER_PIN, LOW);  // Stop cooling, already too cold
    digitalWrite(FAN_PIN, LOW);
    Blynk.virtualWrite(VPIN_SYSTEM_STATE, "Idle (below target)");
    setStatusColor(0, 255, 0);  // Green = stable
  } else {
    digitalWrite(PELTIER_PIN, LOW);
    digitalWrite(FAN_PIN, LOW);
    Blynk.virtualWrite(VPIN_SYSTEM_STATE, "Stable");
    setStatusColor(0, 255, 0);  // Green = stable
  }

  // --- Alarm logic ---
  bool tempOutOfRange = abs(currentTemp - targetTemp) > TEMP_ALARM_MARGIN;
  bool humOutOfRange = abs(currentHumidity - targetHumidity) > HUM_ALARM_MARGIN;

  if (tempOutOfRange || humOutOfRange) {
    if (!alarmActive) {
      alarmActive = true;
      soundAlarm(true);
      setStatusColor(255, 0, 0);  // Red = alarm
      Blynk.virtualWrite(VPIN_ALARM, 1);
      Blynk.logEvent("temp_alarm", "Storage conditions out of safe range!");
    }
  } else {
    if (alarmActive) {
      alarmActive = false;
      soundAlarm(false);
      Blynk.virtualWrite(VPIN_ALARM, 0);
    }
  }

  // --- Battery monitoring ---
  float battVoltage = readBatteryVoltage();
  Blynk.virtualWrite(VPIN_BATTERY, battVoltage);
  if (battVoltage < 10.5) {  // Example low-voltage threshold for a 12V pack
    Blynk.logEvent("low_battery", "Battery voltage critically low!");
  }

  // Debug output
  Serial.printf("Temp: %.1f C | Hum: %.1f%% | Target: %.1f C / %.1f%% | Batt: %.2fV\n",
                currentTemp, currentHumidity, targetTemp, targetHumidity, battVoltage);
}

// ---------------------- Setup ----------------------
void setup() {
  Serial.begin(115200);

  pinMode(PELTIER_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_BLUE_PIN, OUTPUT);
  pinMode(VOLT_SENSOR_PIN, INPUT);

  digitalWrite(PELTIER_PIN, LOW);
  digitalWrite(FAN_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  dht.begin();

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  // Run control loop every 5 seconds
  timer.setInterval(5000L, controlLoop);
}

// ---------------------- Main Loop ----------------------
void loop() {
  Blynk.run();
  timer.run();
}