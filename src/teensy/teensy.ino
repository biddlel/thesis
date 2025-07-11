#include <Audio.h>
#include <arm_math.h>
#include "linalg.h"
#include "AudioFilterSPH0645.h"

// --- Configuration Constants ---
const float SAMPLE_RATE = AUDIO_SAMPLE_RATE_EXACT;
const int FFT_SIZE = 256;
const int NUM_BLOCKS_TO_COLLECT = FFT_SIZE / AUDIO_BLOCK_SAMPLES;
const float SOUND_SPEED = 343000.0; // in mm/s

// --- Microphone Array Setup ---
// TODO
const int NUM_MICS = 4;
struct MicCoordinate { float x, y, z; };
MicCoordinate mic_coords[NUM_MICS] = {
  { 96.41, 90.05, 0.0}, // Mic 1 (top-right)
  {-98.08, 89.86, 0.0}, // Mic 2 (top-left)
  {-99.13,-96.22, 0.0}, // Mic 3 (bottom-left)
  { 96.84,-96.78, 0.0 }  // Mic 4 (bottom-right)
};

// --- MUSIC Algorithm Parameters ---
const int NUM_SOURCES = 1;
const int START_FREQ_BIN = 5;     // Freq bin to start analysis (avoid DC)
const int END_FREQ_BIN = 40;      // Freq bin to end analysis

// --- Localization Search Grid Configuration ---
const float TARGET_PLANE_Z = -500.0; // Z-height of the plane to search (mm)
const float GRID_X_MIN = -1000.0;    // Search area min X (mm)
const float GRID_X_MAX =  1000.0;    // Search area max X (mm)
const int   GRID_X_STEPS = 100;       // Number of steps in X direction
const float GRID_Y_MIN = -1000.0;    // Search area min Y (mm)
const float GRID_Y_MAX =  1000.0;    // Search area max Y (mm)
const int   GRID_Y_STEPS = 100;       // Number of steps in Y direction

// --- Audio System Setup ---
AudioInputI2SQuad    i2s_quad;
AudioFilterSPH0645   mic_correction[NUM_MICS];
AudioRecordQueue     queues[NUM_MICS];
AudioConnection*     patchCords[NUM_MICS];
AudioConnection*     correctionConnection[NUM_MICS];

// --- Buffers and Variables ---
int16_t mic_buffers[NUM_MICS][FFT_SIZE];
int blocks_collected = 0;
float32_t fft_inputs[NUM_MICS][FFT_SIZE];
float32_t fft_outputs[NUM_MICS][FFT_SIZE];  // Only need N/2+1 complex numbers for real FFT
arm_rfft_fast_instance_f32 fft_inst;
float music_spectrum[GRID_X_STEPS][GRID_Y_STEPS];

void setup() {
  Serial.begin(9600);
  Serial1.begin(115200);
  while (!Serial && millis() < 4000) {}

  // Start the audio queues
  AudioMemory(120);
  // Serial.print("Audio memory allocated: ");
  // Serial.println(AudioMemoryUsageMax());
  
  for(int i=0; i<NUM_MICS; ++i) {
    correctionConnection[i] = new AudioConnection(i2s_quad, i, mic_correction[i], 0);
    patchCords[i] = new AudioConnection(mic_correction[i], 0, queues[i], 0);
    queues[i].begin();
    // Serial.print("Started queue for mic ");
    // Serial.println(i);
  }
  arm_rfft_fast_init_f32(&fft_inst, FFT_SIZE);
  // Serial.println("Setup complete. Starting MUSIC coordinate search...");
  delay(1000); // Give some time for serial to initialize
}

void loop() {
  bool all_queues_ready = true;
  // static unsigned long last_print = 0;
  // if (millis() - last_print > 1000) {
  //   last_print = millis();
  //   Serial.print("Queue status: ");
  //   for(int i=0; i<NUM_MICS; ++i) {
  //     Serial.print("[");
  //     Serial.print(queues[i].available());
  //     Serial.print("] ");
  //   }
  //   Serial.println();
  // }
  
  for(int i=0; i<NUM_MICS; ++i) {
    if(queues[i].available() == 0) { all_queues_ready = false; break; }
  }

  if (all_queues_ready) {
    // Serial.println("All queues ready, collecting blocks...");
    // Serial.print("Blocks collected: ");
    // Serial.println(blocks_collected);
    for(int i=0; i<NUM_MICS; ++i) {
      int16_t *data = queues[i].readBuffer();
      if (data) {  // Check if data is valid
        memcpy(mic_buffers[i] + (blocks_collected * AUDIO_BLOCK_SAMPLES), data, AUDIO_BLOCK_SAMPLES * sizeof(int16_t));
        queues[i].freeBuffer();
        // Serial.print("  Mic ");
        // Serial.print(i);
        // Serial.println(" data copied");
      } else {
        // Serial.print("  No data from mic ");
        // Serial.println(i);
      }
    }
    blocks_collected++;

    if (blocks_collected >= NUM_BLOCKS_TO_COLLECT) {
      // Serial.println("\n--- Starting MUSIC algorithm ---");
      run_music_algorithm();
      blocks_collected = 0;
      // Serial.println("--- MUSIC algorithm completed ---\n");
    }
  }
}

void run_music_algorithm() {
  // Serial.println("1. Starting FFT processing...");
  // 1. Perform FFT on all microphone buffers
  for (int i = 0; i < NUM_MICS; ++i) {
    // Apply window function and convert to float32
    for (int j = 0; j < FFT_SIZE; ++j) {
      float32_t window = 0.5f * (1.0f - arm_cos_f32(2 * PI * j / (FFT_SIZE - 1)));
      fft_inputs[i][j] = (float32_t)mic_buffers[i][j] * window;
    }
    // Perform real FFT
    arm_rfft_fast_f32(&fft_inst, fft_inputs[i], fft_outputs[i], 0);  // 0 for forward FFT
    // Serial.print("  FFT completed for mic ");
    // Serial.println(i);
  }

  // 2. Clear the broadband music spectrum
  // Serial.println("2. Clearing music spectrum...");
  for(int ix=0; ix<GRID_X_STEPS; ++ix) for(int iy=0; iy<GRID_Y_STEPS; ++iy) music_spectrum[ix][iy] = 0.0;

  // 3. Iterate over frequency bins to build the broadband spectrum
  for (int k = START_FREQ_BIN; k < END_FREQ_BIN; ++k) {
    float freq_hz = (float)k * SAMPLE_RATE / FFT_SIZE;
    if (freq_hz < 100) continue;

    // 4. Construct and decompose the Spatial Correlation Matrix (SCM)
    Complex scm[NUM_MICS][NUM_MICS];
    for (int i = 0; i < NUM_MICS; ++i) {
      for (int j = 0; j < NUM_MICS; ++j) {
        Complex val_i(fft_outputs[i][2*k], fft_outputs[i][2*k+1]);
        Complex val_j_conj(fft_outputs[j][2*k], -fft_outputs[j][2*k+1]);
        scm[i][j] = val_i * val_j_conj;
      }
    }
    float32_t eigenvalues[NUM_MICS];
    Complex eigenvectors[NUM_MICS][NUM_MICS];
    eigen_decomposition_hermitian_4x4(scm, eigenvalues, eigenvectors);

    // 5. Compute MUSIC pseudospectrum for this frequency bin by searching the grid
    for (int ix = 0; ix < GRID_X_STEPS; ++ix) {
      for (int iy = 0; iy < GRID_Y_STEPS; ++iy) {
        float current_x = GRID_X_MIN + (float)ix * (GRID_X_MAX - GRID_X_MIN) / (GRID_X_STEPS - 1);
        float current_y = GRID_Y_MIN + (float)iy * (GRID_Y_MAX - GRID_Y_MIN) / (GRID_Y_STEPS - 1);
        
        // 6. Compute steering vector for this grid point (spherical wave model)
        Complex steering_vector[NUM_MICS];
        float dx0 = current_x - mic_coords[0].x, dy0 = current_y - mic_coords[0].y, dz0 = TARGET_PLANE_Z - mic_coords[0].z;
        float dist0 = sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0);

        for(int mic=0; mic<NUM_MICS; ++mic) {
          float dxi = current_x - mic_coords[mic].x, dyi = current_y - mic_coords[mic].y, dzi = TARGET_PLANE_Z - mic_coords[mic].z;
          float disti = sqrt(dxi*dxi + dyi*dyi + dzi*dzi);
          float time_diff = (disti - dist0) / SOUND_SPEED;
          float phase_delay = 2.0f * PI * freq_hz * time_diff;
          arm_sin_cos_f32(phase_delay, &steering_vector[mic].im, &steering_vector[mic].re);
          steering_vector[mic].im *= -1; // exp(-j*theta)
        }

        // 7. Calculate pseudospectrum value and add to the grid
        Complex total_proj(0,0);
        for (int n = 0; n < NUM_MICS - NUM_SOURCES; ++n) { // Project onto noise subspace
          Complex proj(0,0);
          for(int mic=0; mic<NUM_MICS; ++mic) {
            proj = proj + steering_vector[mic].conj() * eigenvectors[mic][n];
          }
          total_proj = total_proj + proj * proj.conj();
        }
        if(total_proj.re > 1e-9) music_spectrum[ix][iy] += 1.0f / total_proj.re;
      }
    }
  }

  // 8. Find the peak in the 2D spectrum
  // Serial.println("8. Finding peak in spectrum...");
  float max_val = -1.0; int max_ix = -1, max_iy = -1;
  for (int ix = 0; ix < GRID_X_STEPS; ix++) {
    for (int iy = 0; iy < GRID_Y_STEPS; iy++) {
      if (music_spectrum[ix][iy] > max_val) {
        max_val = music_spectrum[ix][iy];
        max_ix = ix; max_iy = iy;
      }
    }
  }
  
  // 9. Convert peak index back to coordinates
  // Serial.println("9. Converting peak to coordinates...");
  float est_x = GRID_X_MIN + (float)max_ix * (GRID_X_MAX - GRID_X_MIN) / (GRID_X_STEPS - 1);
  float est_y = GRID_Y_MIN + (float)max_iy * (GRID_Y_MAX - GRID_Y_MIN) / (GRID_Y_STEPS - 1);
  
  // Debug: Print some audio samples to verify we're getting data
  // Serial.println("Sample audio data (first 5 samples from each mic):");
  // for(int i=0; i<NUM_MICS; ++i) {
  //   Serial.print("Mic ");
  //   Serial.print(i);
  //   Serial.print(": ");
  //   for(int j=0; j<5; ++j) {
  //     Serial.print(mic_buffers[i][j]);
  //     Serial.print(" ");
  //   }
  //   Serial.println();
  // }

  Serial.print("Estimated Coords (mm): X=");
  Serial.print(est_x, 1);
  Serial.print(", Y=");
  Serial.println(est_y, 1);

  Serial1.print("Estimated Coords (mm): X=");
  Serial1.print(est_x, 1);
  Serial1.print(", Y=");
  Serial1.println(est_y, 1);
}
