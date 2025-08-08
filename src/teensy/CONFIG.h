#pragma once

#define SPEED_OF_SOUND      343.0f      // m/s @ 20°C
#define SAMPLE_RATE_HZ      44100
#define FFT_SIZE            1024        // AudioAnalyzeFFT1024
#define NUM_MICS            4
#define BLOCK_SAMPLES       128         // Teensy Audio block length
#define HISTORY_BLOCKS      (FFT_SIZE / BLOCK_SAMPLES)
#define DEG2RAD             0.017453292519943295f
#define MODE_TIME_SEC       1 // in seconds
#define DOA_OFFSET_DEG      0.0f
#define MIN_FREQ_HZ         800.0f 
#define TOP_MODES           5

static const float micPos[NUM_MICS][3] = {
  { 96.41, 90.05, 0.0}, // Mic 1 (top-right)
  {-98.08, 89.86, 0.0}, // Mic 2 (top-left)
  {-99.13,-96.22, 0.0}, // Mic 3 (bottom-left)
  { 96.84,-96.78, 0.0 }  // Mic 4 (bottom-right)
};

