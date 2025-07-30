#include <Arduino.h>

#include <Audio.h>

#include "config.h"
#include "AudioFilterSPH0645.h"
#include "doa_music.h"

// ──────────────────────────────────────────────────────────────
// Audio design
// ──────────────────────────────────────────────────────────────
AudioInputI2SQuad        i2s1;        // 4‑channel I2S
DOAMusic                 doa;            // Custom AudioStream node
AudioFilterSPH0645       f0, f1, f2, f3;
// AudioFilterBiquad        b0, b1, b2, b3;

AudioConnection          patchCord0(i2s1, 0, f0, 0); // MIC0
AudioConnection          patchCord1(i2s1, 1, f1, 0); // MIC1
AudioConnection          patchCord2(i2s1, 2, f2, 0); // MIC2
AudioConnection          patchCord3(i2s1, 3, f3, 0); // MIC3


AudioConnection          patchCord8(f0, 0, doa, 0); // MIC0
AudioConnection          patchCord9(f1, 0, doa, 1); // MIC1
AudioConnection          patchCord10(f2, 0, doa, 2); // MIC2
AudioConnection          patchCord11(f3, 0, doa, 3); // MIC3

// Mode window histogram
static uint16_t hist[360] = {0};
static uint32_t t0 = 0;

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);
  delay(1000);

  AudioMemory(160);              // plenty for 4 channels + 1 custom node

  doa.begin();
  // windowStartMs = millis();
  // memset(azHist, 0, sizeof(azHist));
  Serial.println(F("DOA_MUSIC started…"));
}

void loop() {
  if (doa.available()) {
    float az = doa.read();                       // 0-359°
    int bin  = int(az + 0.5f) % 360;
    hist[bin]++;

    // end-of-window?
    if (millis() - t0 >= MODE_TIME_SEC * 1000UL) {
      t0 = millis();

      // find TOP_MODES bins
      struct Bin { uint16_t count; uint16_t ang; };
      Bin top[TOP_MODES] = {{0,0}};
      for (int a = 0; a < 360; ++a) {
        uint16_t c = hist[a];
        hist[a] = 0;                     // reset for next window
        // insert into top[] if higher than current minimum
        for (int i = 0; i < TOP_MODES; ++i) {
          if (c > top[i].count) {
            // shift lower entries down
            for (int j = TOP_MODES - 1; j > i; --j) top[j] = top[j - 1];
            top[i] = {c, (uint16_t)a};
            break;
          }
        }
      }

      // serial print
      Serial.print("Angles:");
      Serial1.print("Angles:");
      for (int i = 0; i < TOP_MODES; ++i) {
        if (top[i].count == 0) break;   // fewer than TOP_MODES bins hit
        Serial.print("\t");
        Serial1.print("\t");
        Serial.print(top[i].ang);
        Serial1.print(top[i].ang);
        Serial.print("°:");
        Serial1.print("°:");
        Serial.print(top[i].count);
        Serial1.print(top[i].count);
      }
      Serial.println();
      Serial1.println();
    }
  }
}