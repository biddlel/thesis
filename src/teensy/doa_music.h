/*
 * doa_music.h
 * MUSIC‑based Direction of Arrival estimator (azimuth only)
 *
 * Lightweight 4×4 covariance implementation tuned for Teensy 4.x
 */
#pragma once
#include <Arduino.h>     // ensures stdint types
#include <Audio.h>       // AudioStream & block defs
#include <arm_math.h>
#include "config.h"

class DOAMusic : public AudioStream {
public:
  DOAMusic();
  void begin();
  bool available() const { return _newResult; }
  float read();                 // returns azimuth in degrees
  float lastConfidence() const { return _confidence; }

  virtual void update() override;

private:
  void buildCovariance();       // fill _R (4×4) from buffered samples
  void eigenNoiseSubspace();    // Jacobi eigen‑solve (noise subspace)
  float scanAzimuth();          // MUSIC spectrum search 0‑359°
  inline void clearFlags() { _newResult = false; }

  // state
  bool   _newResult = false;
  float  _azimuthDeg = 0.0f;
  float  _confidence = 0.0f;

  // sample history (float32)
  float  _hist[NUM_MICS][FFT_SIZE];

  // counters
  uint16_t _blockCount = 0;

  // covariance & workspace
  float32_t _R[NUM_MICS * NUM_MICS];       // 4×4 covariance matrix
  float32_t _eigVec[NUM_MICS * NUM_MICS];  // eigenvectors
  float32_t _eigVal[NUM_MICS];             // eigenvalues

  // CMSIS FFT
  arm_rfft_fast_instance_f32 _rfft;

  // input queue for AudioStream
  audio_block_t* _inputQueue[NUM_MICS];
};
