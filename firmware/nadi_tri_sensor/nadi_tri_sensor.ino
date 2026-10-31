/*
 * nadi_tri_sensor.ino  -  ESP32 + TCA9548A + 3x MAX30102 + 3x FSR402
 *
 * Streams one CSV line per tick (OUTPUT_HZ, default 100 Hz):
 *   t_us,ir_vata,red_vata,ir_pitta,red_pitta,ir_kapha,red_kapha,fsr_vata,fsr_pitta,fsr_kapha
 * Lines starting with '#' are status/comments and are ignored by the Python side
 * (scripts/record_session.py, pulse_core/io.py).
 *
 * Timing: every MAX30102 runs at 400 sps with 2x on-chip averaging (200 sps FIFO
 * output). Each 10 ms tick drains every FIFO and averages the ~2 new samples, so
 * all six channels share one timestamp and nothing is aliased or repeated.
 *
 * Auto-gain: at start-up (and on 'g' over serial) each sensor's LED current is
 * stepped until its IR DC level sits in [DC_TARGET_LO, DC_TARGET_HI]. This adapts
 * the optics to each person's skin tone / tissue depth / strap pressure, then
 * LOCKS so the gain never changes mid-recording (that would add steps).
 *
 * Wiring (change the constants below to match your harness):
 *   I2C SDA=21, SCL=22 -> TCA9548A @0x70, MAX30102 (0x57) on mux channels 0,1,2
 *   FSR402: 3V3 -- FSR -- pin -- 10k -- GND   (more force -> higher reading)
 *   Default FSR pins are ADC1 (34, 35, 32) so they keep working if WiFi is enabled.
 *   If your FSRs go through an ADS1115 instead, uncomment USE_ADS1115.
 *
 * Libraries: "SparkFun MAX3010x Pulse and Proximity Sensor Library"
 *            (+ "Adafruit ADS1X15" only when USE_ADS1115 is defined)
 *
 * NOTE: compiles for "ESP32 Dev Module" (esp32 core 3.3.11, both FSR variants) but
 * has not been run on the hardware yet - check the first '# sensor ...' lines of
 * output before recording.
 */
#include <Wire.h>
#include "MAX30105.h"

// #define USE_ADS1115
#ifdef USE_ADS1115
#include <Adafruit_ADS1X15.h>
Adafruit_ADS1115 ads;
#endif

// ---------------- configuration ----------------
const uint32_t BAUD = 115200;          // 100 Hz x ~75 chars fits; raise to 460800 for 200 Hz
const uint32_t OUTPUT_HZ = 100;
const int SDA_PIN = 21, SCL_PIN = 22;
const uint8_t TCA_ADDR = 0x70;
const uint8_t MUX_CH[3] = {0, 1, 2};   // Vata (index), Pitta (middle), Kapha (ring)
const int FSR_PIN[3] = {34, 35, 32};   // ESP32 ADC1 pins (ignored with USE_ADS1115)
const uint8_t FSR_ADS_CH[3] = {0, 1, 2};

// LED current: 0x0A ~ 2 mA ... 0x1F ~ 6 mA ... 0xFF ~ 50 mA.
// v1 crashed at 0x3F with three sensors (supply brown-out), so the AGC is capped.
// Lower LED_MAX if the board resets; raise it only with a stronger 3V3 supply.
const uint8_t LED_START = 0x0A;
const uint8_t LED_MAX = 0x24;
const uint8_t LED_MIN = 0x02;
const uint32_t DC_TARGET_LO = 50000;   // 18-bit ADC, full scale 262143
const uint32_t DC_TARGET_HI = 150000;
// -----------------------------------------------

MAX30105 sensor[3];
bool sensorOk[3] = {false, false, false};
uint8_t ledAmp[3] = {LED_START, LED_START, LED_START};
uint32_t lastIr[3] = {0, 0, 0}, lastRed[3] = {0, 0, 0};
const uint32_t TICK_US = 1000000UL / OUTPUT_HZ;
uint32_t nextTick = 0;

void tcaSelect(uint8_t ch) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

void setLed(int i, uint8_t amp) {
  tcaSelect(MUX_CH[i]);
  sensor[i].setPulseAmplitudeIR(amp);
  sensor[i].setPulseAmplitudeRed(amp);
  sensor[i].setPulseAmplitudeGreen(0);
}

// Drain the FIFO and average all new samples. Returns number of samples read.
int readAveraged(int i, uint32_t &ir, uint32_t &red) {
  tcaSelect(MUX_CH[i]);
  sensor[i].check();
  uint64_t sIr = 0, sRed = 0;
  int n = 0;
  while (sensor[i].available()) {
    // SparkFun's names follow the MAX30105; on many MAX30102 boards Red/IR come out
    // swapped. IR should have the larger DC; swap here if yours is reversed.
    sIr += sensor[i].getFIFOIR();
    sRed += sensor[i].getFIFORed();
    sensor[i].nextSample();
    n++;
  }
  if (n > 0) {
    lastIr[i] = sIr / n;
    lastRed[i] = sRed / n;
  }
  ir = lastIr[i];
  red = lastRed[i];
  return n;
}

uint16_t readFsr(int i) {
#ifdef USE_ADS1115
  int16_t v = ads.readADC_SingleEnded(FSR_ADS_CH[i]);
  return v < 0 ? 0 : (uint16_t)v;
#else
  uint32_t s = 0;
  for (int k = 0; k < 4; k++) s += analogRead(FSR_PIN[i]);
  return s / 4;
#endif
}

void autoGain() {
  Serial.println("# auto-gain: keep the band on the wrist and stay still...");
  for (int i = 0; i < 3; i++) {
    if (!sensorOk[i]) continue;
    uint8_t amp = LED_START;
    for (int iter = 0; iter < 24; iter++) {
      setLed(i, amp);
      delay(60);
      uint32_t ir, red, acc = 0;
      int got = 0;
      for (int k = 0; k < 10; k++) {
        delay(10);
        if (readAveraged(i, ir, red) > 0) { acc += ir; got++; }
      }
      uint32_t dc = got ? acc / got : 0;
      if (dc < DC_TARGET_LO && amp < LED_MAX) amp = min<int>(LED_MAX, amp + 2);
      else if (dc > DC_TARGET_HI && amp > LED_MIN) amp = max<int>(LED_MIN, amp - 2);
      else break;
    }
    ledAmp[i] = amp;
    setLed(i, amp);
  }
  Serial.printf("# led_amp=0x%02X,0x%02X,0x%02X  (locked)\n", ledAmp[0], ledAmp[1], ledAmp[2]);
}

void setup() {
  Serial.begin(BAUD);
  delay(1000);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

#ifdef USE_ADS1115
  if (!ads.begin()) Serial.println("# [FAIL] ADS1115 not found at 0x48");
  ads.setGain(GAIN_ONE);                 // +-4.096 V
  ads.setDataRate(RATE_ADS1115_860SPS);  // ~1.2 ms per read, 3 reads fit in a 10 ms tick
#else
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);        // ~0-3.3 V range
#endif

  if (!i2cPresent(TCA_ADDR)) {
    Serial.println("# [FAIL] TCA9548A not found at 0x70 - check wiring");
    while (true) delay(1000);
  }
  for (int i = 0; i < 3; i++) {
    tcaSelect(MUX_CH[i]);
    delay(5);
    if (!i2cPresent(0x57)) {
      Serial.printf("# [FAIL] sensor %d: no MAX30102 on mux channel %d\n", i, MUX_CH[i]);
      continue;
    }
    // begin() also checks the part ID; some clone boards report a wrong ID but
    // work fine, so a false return is only a warning.
    if (!sensor[i].begin(Wire, I2C_SPEED_FAST)) Serial.printf("# [WARN] sensor %d: unexpected part ID\n", i);
    // power, sampleAverage, ledMode(2=Red+IR), sampleRate, pulseWidth(us), adcRange(nA)
    sensor[i].setup(LED_START, 2, 2, 400, 411, 4096);
    setLed(i, LED_START);
    sensorOk[i] = true;
    Serial.printf("# sensor %d OK on mux channel %d\n", i, MUX_CH[i]);
  }
  autoGain();
  Serial.printf("# fs=%lu\n", (unsigned long)OUTPUT_HZ);
  Serial.println("t_us,ir_vata,red_vata,ir_pitta,red_pitta,ir_kapha,red_kapha,fsr_vata,fsr_pitta,fsr_kapha");
  nextTick = micros();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'g') autoGain();
  }

  while ((int32_t)(micros() - nextTick) < 0) {}
  uint32_t t = nextTick;
  nextTick += TICK_US;

  uint32_t ir[3], red[3];
  for (int i = 0; i < 3; i++) {
    if (sensorOk[i]) readAveraged(i, ir[i], red[i]);
    else ir[i] = red[i] = 0;  // 0 = "missing", the Python side masks it
  }
  uint16_t f[3];
  for (int i = 0; i < 3; i++) f[i] = readFsr(i);

  Serial.printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,%u,%u,%u\n", (unsigned long)t,
                (unsigned long)ir[0], (unsigned long)red[0], (unsigned long)ir[1], (unsigned long)red[1],
                (unsigned long)ir[2], (unsigned long)red[2], f[0], f[1], f[2]);

  // if we fell behind (e.g. after auto-gain), resynchronise instead of bursting
  if ((int32_t)(micros() - nextTick) > (int32_t)(5 * TICK_US)) nextTick = micros();
}
