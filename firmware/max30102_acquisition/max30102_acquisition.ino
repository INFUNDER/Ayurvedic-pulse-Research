#include <Wire.h>
#include "MAX30105.h"

#define SDA_PIN 21
#define SCL_PIN 22
#define TCA_ADDR 0x70

MAX30105 sensor1;
MAX30105 sensor2;
MAX30105 sensor3;

MAX30105* sensors[3] = { &sensor1, &sensor2, &sensor3 };
uint8_t channels[3] = { 0, 1, 2 };

void selectChannel(uint8_t channel) {
  if (channel > 7) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200); // Reverted back to 115200 for Mac driver compatibility
  delay(1000);
  
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000); // 400kHz Fast I2C mode
  delay(500);

  // Initialize all 3 sensors exactly like your test script
  for (int i = 0; i < 3; i++) {
    selectChannel(channels[i]);
    delay(100);

    // Bypassing the safety check! We call begin() to give the library the I2C port,
    // but we completely IGNORE if it returns false due to a zombie Part ID.
    sensors[i]->begin(Wire, I2C_SPEED_FAST);
    
    // High-speed hardware configuration
    sensors[i]->setup(
      60,     // LED brightness (not used since we override below)
      1,      // Sample averaging (1 = NO averaging, highest raw speed)
      2,      // LED mode: Red + IR
      400,    // Sample rate (Max speed for SpO2 mode)
      411,    // Pulse width
      4096    // ADC range
    );

    // CRITICAL FIX: Extremely low power (0x0A) instead of (0x3F) to prevent crashing
    sensors[i]->setPulseAmplitudeRed(0x0A);
    sensors[i]->setPulseAmplitudeIR(0x0A);
    sensors[i]->setPulseAmplitudeGreen(0);
    delay(100);
  }
}

void loop() {
  long irValues[3] = {0, 0, 0};

  for (int i = 0; i < 3; i++) {
    selectChannel(channels[i]);
    irValues[i] = sensors[i]->getIR();
  }

  // CRITICAL FIX: Format exactly as Python expects
  Serial.print("Vata:");
  Serial.print(irValues[0]);
  Serial.print(",Pitta:");
  Serial.print(irValues[1]);
  Serial.print(",Kapha:");
  Serial.println(irValues[2]);

  // Removed delay(10) to allow the ESP32 to loop as fast as the sensors can output
}