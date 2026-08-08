/*
 * PrakritiAI - Nadi Pariksha Data Collection
 * Reads raw IR waveforms from two MAX30102 sensors via a TCA9548A Multiplexer.
 * Formatted for the Arduino Serial Plotter to visualize Vata and Pitta waveforms.
 */

#include <Wire.h>
#include "MAX30105.h" // This library covers the MAX30102

#define TCAADDR 0x70  // Default I2C address for the TCA9548A

// Create sensor objects for our specific Nadi positions
MAX30105 sensorVata;  // Connected to Multiplexer Channel 0
MAX30105 sensorPitta; // Connected to Multiplexer Channel 1
MAX30105 sensorKapha; // Connected to Multiplexer Channel 2

// Helper function to switch the multiplexer channel
void tcaselect(uint8_t i) {
  if (i > 7) return; 
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();  
}

void setup() {
  Serial.begin(115200);
  Wire.begin(); // Join the main I2C bus

  // --------------------------------------------------
  // Initialize Vata Sensor (Channel 0)
  // --------------------------------------------------
  tcaselect(0); 
  if (!sensorVata.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Vata sensor not found on Channel 0. Check wiring.");
    while (1); // Halt execution if missing
  }
  sensorVata.setup(); 
  // Turn off Red and Green LEDs. We only need the IR LED for deep tissue pulse morphology.
  sensorVata.setPulseAmplitudeRed(0x0A); 
  sensorVata.setPulseAmplitudeGreen(0x0A); 

  // --------------------------------------------------
  // Initialize Pitta Sensor (Channel 1)
  // --------------------------------------------------
  tcaselect(1); 
  if (!sensorPitta.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Pitta sensor not found on Channel 1. Check wiring.");
    while (1); // Halt execution if missing
  }
  sensorPitta.setup(); 
  sensorPitta.setPulseAmplitudeRed(0x0A); 
  sensorPitta.setPulseAmplitudeGreen(0x0A);

  // --------------------------------------------------
  // Initialize Kapha Sensor (Channel 2)
  // --------------------------------------------------
  tcaselect(2); 
  if (!sensorKapha.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Kapha sensor not found on Channel 2. Check wiring.");
    while (1); // Halt execution if missing
  }
  sensorKapha.setup(); 
  sensorKapha.setPulseAmplitudeRed(0x0A); 
  sensorKapha.setPulseAmplitudeGreen(0x0A);
}

void loop() {
  // Read raw IR value from the Vata position
  tcaselect(0); 
  long irVata = sensorVata.getIR();
  
  // Read raw IR value from the Pitta position
  tcaselect(1); 
  long irPitta = sensorPitta.getIR();

  // Read raw IR value from the Kapha position
  tcaselect(2); 
  long irKapha = sensorKapha.getIR();

  // Print the data in a format the Arduino Serial Plotter understands:
  // "Vata:Value1,Pitta:Value2,Kapha:Value3"
  Serial.print("Vata:");
  Serial.print(irVata);
  Serial.print(",Pitta:");
  Serial.print(irPitta);
  Serial.print(",Kapha:");
  Serial.println(irKapha);

  // A tiny delay keeps the Serial Plotter graph moving at a readable speed
  delay(10); 
}