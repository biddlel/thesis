/* SPH0645 MEMS Microphone Test (Adaf/Users/bids/Library/CloudStorage/OneDrive-UNSW/thesis/src/teensy/sos-iir-filter.hruit product #3421)
 *
 * Forum thread with connection details and other info:
 * https://forum.pjrc.com/threads/60599?p=238070&viewfull=1#post238070
*/


#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

#include "AudioFilterAWeight.h"
#include "AudioFilterSPH0645.h"
// #include "input_i2s_32.h"

const float FS = AUDIO_SAMPLE_RATE;                   // Sampling rate
const float REF_PRESSURE = 0.00002f;

const float MIC_SENS_dBV = -29.0;    // Sensitivity in dBV
const float MIC_REF_dB = 94.0;       // Reference SPL for mic sensitivity (dB SPL)
const float VREF = 3.3f;             // Teensy analog ref voltage
const float ADC_FULL_SCALE = 1.0f;   // Full scale is normalized to 1.0
const float MIC_SENS_CORRECTION_dB = 94.0f + MIC_SENS_dBV;  // Correction factor

float sumSquares = 0.0;
unsigned long sampleCount = 0;
unsigned long lastUpdate = 0;
const unsigned long UPDATE_INTERVAL_MS = 1000;

AudioInputI2S       i2s1;           //xy=180,111
AudioFilterSPH0645  micCorrection;
AudioFilterAWeight  aWeight;
AudioAnalyzeRMS     rms;
AudioAnalyzePeak    peak;   
// 
AudioConnection patchCord1(i2s1, 0, micCorrection, 0);
AudioConnection patchCord2(micCorrection, aWeight);
AudioConnection patchCord3(aWeight, peak);

// Fast time window (125 ms)
const unsigned long FAST_WINDOW_MS = 125;

// To accumulate block-by-block power
float   sumPower    = 0.0f;   // sum of (blockRMS)^2
uint16_t blockCount = 0;      // how many blocks were summed
unsigned long windowStartMs;


void setup() {
  AudioMemory(12);
  lastUpdate = millis();
}

void loop() {
  if (peak.available() &&  (millis() - lastUpdate) >= FAST_WINDOW_MS) {
    // 1. Read A-weighted RMS (unitless, 0…1)
    float blockRMS = peak.read();
    // 2. Convert to dBFS:
    float dBFS_rms = 20.0f * log10f(blockRMS);
    // 3. Convert to dB SPL via calibration
    float dBSPL = dBFS_rms + MIC_SENS_CORRECTION_dB;
    // 4. Print it
    Serial.print("blockRMS = ");
    Serial.print(blockRMS, 6);
    Serial.print("   dBFS = ");
    Serial.print(dBFS_rms, 2);
    Serial.print("   dB SPL = ");
    Serial.println(dBSPL, 2);
    lastUpdate = millis();
  }
}
