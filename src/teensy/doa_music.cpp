/*
 * doa_music.cpp – implementation
 */
#include "doa_music.h"

#define TWO_PI_F 6.28318530718f

// ──────────────────────────────────────────────────────────────
// Forward
// ──────────────────────────────────────────────────────────────
static void jacobi4(float32_t *A, float32_t *V, float32_t *d);

DOAMusic::DOAMusic()
: AudioStream(NUM_MICS, _inputQueue) {
  // nothing else
}

void DOAMusic::begin() {
  arm_rfft_fast_init_f32(&_rfft, FFT_SIZE);
}

float DOAMusic::read() {
  _newResult = false;
  return _azimuthDeg;
}

// ──────────────────────────────────────────────────────────────
// Audio ISR (128 sample blocks)
// ──────────────────────────────────────────────────────────────
void DOAMusic::update() {
  audio_block_t *in[NUM_MICS];
  for (int ch = 0; ch < NUM_MICS; ++ch) {
    in[ch] = receiveReadOnly(ch);
  }
  if (!in[0]) return;

  const float scale = 1.0f / 32768.0f; // int16_t → ±1
  uint16_t offset = (_blockCount % HISTORY_BLOCKS) * BLOCK_SAMPLES;

  for (int ch = 0; ch < NUM_MICS; ++ch) {
    float *dst = &_hist[ch][offset];
    for (int i = 0; i < BLOCK_SAMPLES; ++i) {
      dst[i] = (float)in[ch]->data[i] * scale;
    }
  }

  ++_blockCount;
  if (_blockCount % HISTORY_BLOCKS == 0) {
    buildCovariance();
    eigenNoiseSubspace();
    _azimuthDeg = scanAzimuth();
    _confidence = 1.0f; // TODO confidence metric
    _newResult = true;
  }

  for (int ch = 0; ch < NUM_MICS; ++ch) {
    release(in[ch]);
  }
}

// ──────────────────────────────────────────────────────────────
// Covariance (real‑valued time domain)
// ──────────────────────────────────────────────────────────────
void DOAMusic::buildCovariance() {
  for (int r = 0; r < NUM_MICS; ++r) {
    for (int c = 0; c < NUM_MICS; ++c) {
      float32_t sum = 0.0f;
      for (int n = 0; n < FFT_SIZE; ++n) {
        sum += _hist[r][n] * _hist[c][n];
      }
      _R[r * NUM_MICS + c] = sum / FFT_SIZE;
    }
  }
}

void DOAMusic::eigenNoiseSubspace() {
  jacobi4(_R, _eigVec, _eigVal);
  // sort ascending
  for (int i = 0; i < NUM_MICS - 1; ++i) {
    for (int j = i + 1; j < NUM_MICS; ++j) {
      if (_eigVal[i] > _eigVal[j]) {
        std::swap(_eigVal[i], _eigVal[j]);
        for (int k = 0; k < NUM_MICS; ++k) {
          std::swap(_eigVec[k + i * NUM_MICS], _eigVec[k + j * NUM_MICS]);
        }
      }
    }
  }
}

float DOAMusic::scanAzimuth() {
  const int NUM_STEERS = 360;
  float32_t bestP = 0.0f;
  int bestIdx = 0;

  // Build noise subspace projection matrix EnEnH = En * Enᵀ
  float32_t En[NUM_MICS * (NUM_MICS - 1)];
  for (int c = 0; c < NUM_MICS - 1; ++c) {
    for (int r = 0; r < NUM_MICS; ++r) {
      En[r + c * NUM_MICS] = _eigVec[r + c * NUM_MICS];
    }
  }
  float32_t EnEnH[NUM_MICS * NUM_MICS] = {0};
  for (int r = 0; r < NUM_MICS; ++r) {
    for (int c = 0; c < NUM_MICS; ++c) {
      float32_t s = 0.0f;
      for (int k = 0; k < NUM_MICS - 1; ++k) {
        s += En[r + k * NUM_MICS] * En[c + k * NUM_MICS];
      }
      EnEnH[r * NUM_MICS + c] = s;
    }
  }

  for (int ang = 0; ang < NUM_STEERS; ++ang) {
    float theta = ang * DEG2RAD;

    // steering vector (real, broadband approximation)
    float sv[NUM_MICS];
    for (int m = 0; m < NUM_MICS; ++m) {
      float x = micPos[m][0] * 0.001f;
      float y = micPos[m][1] * 0.001f;
      float proj = x * cosf(theta) + y * sinf(theta);
      float delay = proj / SPEED_OF_SOUND;
      sv[m] = cosf(TWO_PI_F * SAMPLE_RATE_HZ * delay);
    }

    // tmp = svᵀ * EnEnH
    float tmp[NUM_MICS] = {0};
    for (int c = 0; c < NUM_MICS; ++c) {
      for (int r = 0; r < NUM_MICS; ++r) {
        tmp[c] += sv[r] * EnEnH[r * NUM_MICS + c];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < NUM_MICS; ++i) denom += tmp[i] * sv[i];

    float P = 1.0f / (denom + 1e-9f);
    if (P > bestP) { bestP = P; bestIdx = ang; }
  }
  return (float)bestIdx;
}

// ──────────────────────────────────────────────────────────────
// Jacobi eigen decomposition (4×4) – adapted, in‑place
// ──────────────────────────────────────────────────────────────
static void jacobi4(float32_t *A, float32_t *V, float32_t *d) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      V[i*4 + j] = (i == j) ? 1.0f : 0.0f;
    }
    d[i] = A[i*4 + i];
  }
  float32_t B[4], Z[4];
  memcpy(B, d, sizeof(B));
  memset(Z, 0, sizeof(Z));

  for (int iter = 0; iter < 25; ++iter) {
    float32_t sm = 0.0f;
    for (int p = 0; p < 3; ++p)
      for (int q = p+1; q < 4; ++q)
        sm += fabsf(A[p*4 + q]);
    if (sm < 1e-9f) break;

    float32_t tresh = (iter < 3) ? 0.2f * sm / 16.0f : 0.0f;

    for (int p = 0; p < 3; ++p) {
      for (int q = p+1; q < 4; ++q) {
        float32_t g = 100.0f * fabsf(A[p*4 + q]);
        if (iter > 3 && (fabsf(d[p]) + g) == fabsf(d[p])
                    && (fabsf(d[q]) + g) == fabsf(d[q])) {
          A[p*4 + q] = 0.0f;
        } else if (fabsf(A[p*4 + q]) > tresh) {
          float32_t h = d[q] - d[p];
          float32_t t;
          if ((fabsf(h) + g) == fabsf(h)) {
            t = A[p*4 + q] / h;
          } else {
            float32_t theta = 0.5f * h / A[p*4 + q];
            t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta*theta));
            if (theta < 0.0f) t = -t;
          }
          float32_t c = 1.0f / sqrtf(1.0f + t*t);
          float32_t s = t * c;
          float32_t tau = s / (1.0f + c);
          h = t * A[p*4 + q];
          Z[p] -= h;
          Z[q] += h;
          d[p] -= h;
          d[q] += h;
          A[p*4 + q] = 0.0f;

          for (int j = 0; j < p; ++j) {
            float32_t g1 = A[j*4 + p];
            float32_t h1 = A[j*4 + q];
            A[j*4 + p] = g1 - s*(h1 + g1*tau);
            A[j*4 + q] = h1 + s*(g1 - h1*tau);
          }
          for (int j = p+1; j < q; ++j) {
            float32_t g1 = A[p*4 + j];
            float32_t h1 = A[j*4 + q];
            A[p*4 + j] = g1 - s*(h1 + g1*tau);
            A[j*4 + q] = h1 + s*(g1 - h1*tau);
          }
          for (int j = q+1; j < 4; ++j) {
            float32_t g1 = A[p*4 + j];
            float32_t h1 = A[q*4 + j];
            A[p*4 + j] = g1 - s*(h1 + g1*tau);
            A[q*4 + j] = h1 + s*(g1 - h1*tau);
          }
          for (int j = 0; j < 4; ++j) {
            float32_t g1 = V[j*4 + p];
            float32_t h1 = V[j*4 + q];
            V[j*4 + p] = g1 - s*(h1 + g1*tau);
            V[j*4 + q] = h1 + s*(g1 - h1*tau);
          }
        }
      }
    }
    for (int p = 0; p < 4; ++p) {
      B[p] += Z[p];
      d[p] = B[p];
      Z[p] = 0.0f;
    }
  }
}
