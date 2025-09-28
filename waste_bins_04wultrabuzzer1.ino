#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Servos
Servo servoB; // Biodegradable
Servo servoR; // Recyclable
Servo servoS; // Residual

// LCD
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Ultrasonic sensor
const int trigPin = 6;
const int echoPin = 7;
long duration;
int distance;
const int FULL_DISTANCE = 5; // cm, adjust based on your bin

// Buzzer
const int buzzerPin = 8;

// Scrolling message
String message = "   Smart Waste Segregator   "; 

void setup() {
  Serial.begin(9600);

  // Attach servos
  servoB.attach(9);
  servoR.attach(10);
  servoS.attach(11);

  // LCD setup
  lcd.init();
  lcd.backlight();

  // Scroll startup message
  scrollMessage(message, 300);

  // Show ready state
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready...");

  // Ultrasonic pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Buzzer
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW);
}

void loop() {
  // Check bin fullness
  distance = getDistance();
  if (distance > 0 && distance < FULL_DISTANCE) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(" Bin is FULL!");
    tone(buzzerPin, 1000); // 1kHz beep
    delay(1000);
    noTone(buzzerPin);

    delay(2000); // keep full message visible
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("System Ready...");
  }

  // Listen for Python commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'B') handleBin(servoB, "Biodegradable");
    else if (cmd == 'R') handleBin(servoR, "Recyclable");
    else if (cmd == 'S') handleBin(servoS, "Residual");
  }
}

void handleBin(Servo &s, const char *label) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(label);
  lcd.setCursor(0, 1);
  lcd.print("Bin Opens...");

  s.write(90);   // open bin
  delay(2000);

  s.write(0);    // close bin
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Bin Closed");
  delay(2000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready...");
}

int getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH, 20000); // timeout 20ms
  if (duration == 0) return -1; // no echo
  return duration * 0.034 / 2; // convert to cm
}

void scrollMessage(String msg, int delayTime) {
  int msgLength = msg.length();
  for (int i = 0; i < msgLength - 15; i++) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(msg.substring(i, i + 16)); 
    delay(delayTime);
  }
}
