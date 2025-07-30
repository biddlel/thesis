#include <Arduino.h>

#include <Audio.h>

#include "config.h"
#include "doa_music.h"

// ──────────────────────────────────────────────────────────────
// Audio design
// ──────────────────────────────────────────────────────────────
AudioInputI2SQuad        i2s1;        // 4‑channel I2S
DOAMusic                 doa;            // Custom AudioStream node
AudioConnection          patchCord0(i2s1, 0, doa, 0); // MIC0
AudioConnection          patchCord1(i2s1, 1, doa, 1); // MIC1
AudioConnection          patchCord2(i2s1, 2, doa, 2); // MIC2
AudioConnection          patchCord3(i2s1, 3, doa, 3); // MIC3

// Mode window histogram
static uint16_t azHist[360];
uint32_t windowStartMs = 0;

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);
  delay(1000);

  AudioMemory(80);              // plenty for 4 channels + 1 custom node

  doa.begin();
  windowStartMs = millis();
  memset(azHist, 0, sizeof(azHist));
  Serial.println(F("DOA_MUSIC started…"));
}

void loop() {
  if (doa.available()) {
    float az = doa.read();
    int bin = (int)(az + 0.5f) % 360;
    azHist[bin]++;

    uint32_t now = millis();
    if (now - windowStartMs >= MODE_TIME_SEC * 1000UL) {
      // compute mode
      uint16_t maxCount = 0;
      int modeAz = 0;
      for (int i = 0; i < 360; ++i) {
        if (azHist[i] > maxCount) {
          maxCount = azHist[i];
          modeAz = i;
        }
        azHist[i] = 0; // reset for next window
      }
      windowStartMs = now;

      Serial.print("Angle: ");
      Serial.print(modeAz);
      Serial.print(", Count: ");
      Serial.println(maxCount);

      Serial1.print("Angle: ");
      Serial1.print(modeAz);
      Serial1.print(", Count: ");
      Serial1.println(maxCount);
    }
  }
}