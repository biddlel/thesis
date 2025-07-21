#include <Audio.h>
#include <arm_math.h>
#include "linalg.h"
// Simple binary protocol for spectrum data

// Buffer for serial communication
const size_t BUFFER_SIZE = 4096;  // Buffer size for serial communication
uint8_t buffer[BUFFER_SIZE];

// Protocol constants
const uint8_t START_MARKER = 0x1F;
const uint8_t END_MARKER = 0x1E;

#define DEBUGLOG_DEFAULT_LOG_LEVEL DEBUGLOG_LEVEL_WARN
#define ARDUINOJSON_USE_LONG_LONG 1
#define ARDUINOJSON_USE_DOUBLE 1
#define ARDUINOJSON_DECODE_UNICODE 1

// Structure to hold sound source location
struct SoundSource {
    float x;        // X coordinate
    float y;        // Y coordinate
    float strength; // Relative strength (0.0 to 1.0)
};

// Sound sources data structure
struct SoundSources {
    uint8_t count;  // Number of sources (max 255)
    SoundSource sources[10]; // Fixed size array for simplicity
    uint32_t checksum;
};

// Global instance
SoundSources sound_sources_msg;

// Structure to hold spectrum data for sound source detection
struct SpectrumData {
    std::vector<std::vector<float>> spectrum_data;
    
    // Add minimal required members for compatibility
    uint32_t x_steps = 0;
    uint32_t y_steps = 0;
    float x_min = 0, x_max = 0;
    float y_min = 0, y_max = 0;
};

// Global instance
SpectrumData spectrum_msg;

// --- Configuration Constants ---
const float SAMPLE_RATE = AUDIO_SAMPLE_RATE_EXACT;
const int FFT_SIZE = 1024;
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
const int START_FREQ_BIN = 10;     // Freq bin to start analysis (avoid DC)
const int END_FREQ_BIN = 100;      // Freq bin to end analysis
const float NOISE_FLOOR = 1e-6f;    // Small value to avoid division by zero
const float MIN_EIGENVALUE = 1e-10f; // Minimum eigenvalue to consider

const int MAX_SOURCES = 5;

// --- Localization Search Grid Configuration ---
const float GRID_SEARCH = 1000;    // Search radius in mm
const int STEPS = 20;             // Number of steps per dimension (reduced for 3D search)

// 3D Search space boundaries (in mm)
const float GRID_X_MIN = -GRID_SEARCH;    // Min X coordinate
const float GRID_X_MAX =  GRID_SEARCH;    // Max X coordinate
const float GRID_Y_MIN = -GRID_SEARCH;    // Min Y coordinate
const float GRID_Y_MAX =  GRID_SEARCH;    // Max Y coordinate
const float GRID_Z_MIN =  0;          // Min Z coordinate (don't search too close to mics)
const float GRID_Z_MAX = 1000.0;          // Max Z coordinate

// Number of steps in each dimension
const int GRID_X_STEPS = STEPS;
const int GRID_Y_STEPS = STEPS;
const int GRID_Z_STEPS = 4;  // Fewer steps in Z to reduce computation

// --- Audio System Setup ---
AudioInputI2SQuad    i2s_quad;
// AudioFilterSPH0645   mic_correction[NUM_MICS];
AudioRecordQueue     queues[NUM_MICS];
AudioConnection*     patchCords[NUM_MICS];
// AudioConnection*     correctionConnection[NUM_MICS];

// --- Buffers and Variables ---
int16_t mic_buffers[NUM_MICS][FFT_SIZE];
int blocks_collected = 0;
float32_t fft_inputs[NUM_MICS][FFT_SIZE];
float32_t fft_outputs[NUM_MICS][FFT_SIZE];         // FFT outputs for each mic
arm_rfft_fast_instance_f32 fft_inst;

// 3D MUSIC spectrum (using 1D array for better memory management)
float music_spectrum[GRID_X_STEPS * GRID_Y_STEPS * GRID_Z_STEPS];

// Helper function to convert 3D grid indices to 1D array index
inline int gridToIndex(int ix, int iy, int iz) {
  return ix + GRID_X_STEPS * (iy + GRID_Y_STEPS * iz);
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);
  while (!Serial && millis() < 4000) {}

  // Start the audio queues
  AudioMemory(240);
  // Serial.print("Audio memory allocated: ");
  // Serial.println(AudioMemoryUsageMax());
  
  for(int i=0; i<NUM_MICS; ++i) {
    // correctionConnection[i] = new AudioConnection(i2s_quad, i, mic_correction[i], 0);
    patchCords[i] = new AudioConnection(i2s_quad, i, queues[i], 0);
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

// Calculate a simple checksum for the header
uint32_t calculate_checksum(const uint8_t* data, size_t length) {
  uint32_t sum = 0;
  for (size_t i = 0; i < length; i++) {
      sum += data[i];
  }
  return sum;
}

// Simple checksum calculation for sound sources

// Function to detect sound sources from 3D spectrum data
void detect_sound_sources(SoundSources& sources) {
    const float THRESHOLD = 0.5f;  // Minimum intensity threshold (0.0 to 1.0)
    const float MIN_DISTANCE = 0.1f; // Minimum distance between sources (normalized)
    
    // Reset sources
    sources.count = 0;
    
    // Find local maxima in 3D spectrum
    for (int z = 1; z < GRID_Z_STEPS - 1; z++) {
        for (int y = 1; y < GRID_Y_STEPS - 1; y++) {
            for (int x = 1; x < GRID_X_STEPS - 1; x++) {
                float val = music_spectrum[gridToIndex(x, y, z)];
                
                // Skip if below threshold
                if (val < THRESHOLD) continue;
                
                // Check if this is a local maximum in 3D space
                bool is_local_max = true;
                for (int dz = -1; dz <= 1 && is_local_max; dz++) {
                    for (int dy = -1; dy <= 1 && is_local_max; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            if (music_spectrum[gridToIndex(x+dx, y+dy, z+dz)] >= val) {
                                is_local_max = false;
                                break;
                            }
                        }
                    }
                }
                
                if (!is_local_max) continue;
                
                // Convert grid position to coordinates
                float x_pos = GRID_X_MIN + (x / (float)(GRID_X_STEPS - 1)) * (GRID_X_MAX - GRID_X_MIN);
                float y_pos = GRID_Y_MIN + (y / (float)(GRID_Y_STEPS - 1)) * (GRID_Y_MAX - GRID_Y_MIN);
                
                // Check if this source is too close to existing ones
                bool too_close = false;
                for (uint8_t i = 0; i < sources.count; i++) {
                    float dx = x_pos - sources.sources[i].x;
                    float dy = y_pos - sources.sources[i].y;
                    float dist = sqrtf(dx*dx + dy*dy);
                    
                    if (dist < MIN_DISTANCE) {
                        // Keep the stronger source
                        if (val > sources.sources[i].strength) {
                            sources.sources[i].x = x_pos;
                            sources.sources[i].y = y_pos;
                            sources.sources[i].strength = val;
                        }
                        too_close = true;
                        break;
                    }
                }
                
                // Add as new source if not too close to existing ones
                if (!too_close && sources.count < 10) {
                    sources.sources[sources.count].x = x_pos;
                    sources.sources[sources.count].y = y_pos;
                    sources.sources[sources.count].strength = val;
                    sources.count++;
                }
            }
        }
    }
    
    // Sort sources by strength (strongest first)
    for (uint8_t i = 0; i < sources.count; i++) {
        for (uint8_t j = i + 1; j < sources.count; j++) {
            if (sources.sources[i].strength < sources.sources[j].strength) {
                SoundSource temp = sources.sources[i];
                sources.sources[i] = sources.sources[j];
                sources.sources[j] = temp;
            }
        }
    }
    
    // Calculate checksum
    sources.checksum = 0;
    uint8_t* ptr = (uint8_t*)&sources;
    for (size_t i = 0; i < sizeof(SoundSources) - sizeof(uint32_t); i++) {
        sources.checksum += ptr[i];
    }
}

// Write a 32-bit value in big-endian order
void write_uint32(uint8_t* buffer, uint32_t value) {
    buffer[0] = (value >> 24) & 0xFF;
    buffer[1] = (value >> 16) & 0xFF;
    buffer[2] = (value >> 8) & 0xFF;
    buffer[3] = value & 0xFF;
}

// Write a 32-bit float in big-endian order
void write_float(uint8_t* buffer, float value) {
    uint32_t* val_ptr = (uint32_t*)&value;
    write_uint32(buffer, *val_ptr);
}

bool send_sound_sources() {
    // Detect sound sources from the current spectrum
    detect_sound_sources(sound_sources_msg);
    
    // Print all sources in one line
    Serial.print("Sources[");
    Serial.print(sound_sources_msg.count);
    Serial.print("]: ");
    
    for (uint8_t i = 0; i < sound_sources_msg.count && i < MAX_SOURCES; i++) {
        if (i > 0) Serial.print(" | ");
        Serial.print(i);
        Serial.print(":(");
        Serial.print(sound_sources_msg.sources[i].x, 1);
        Serial.print(",");
        Serial.print(sound_sources_msg.sources[i].y, 1);
        Serial.print(",");
        Serial.print(sound_sources_msg.sources[i].strength, 2);
        Serial.print(")");
    }
    Serial.println();
    
    // Add a small delay to make the output readable
    delay(100);
    
    return true;
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

  // 2. Reset the spectrum
  // Serial.println("1. Resetting spectrum...");
  int total_points = GRID_X_STEPS * GRID_Y_STEPS * GRID_Z_STEPS;
  for(int i=0; i<total_points; ++i) {
    music_spectrum[i] = 0.0;
  }
  
  // Initialize spectrum data structure for sound source detection
  spectrum_msg.spectrum_data.resize(GRID_Y_STEPS);
  for (int y = 0; y < GRID_Y_STEPS; y++) {
    spectrum_msg.spectrum_data[y].resize(GRID_X_STEPS);
  }

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

    // 5. Compute MUSIC pseudospectrum for this frequency bin by searching the 3D grid
    for (int ix = 0; ix < GRID_X_STEPS; ++ix) {
      float current_x = GRID_X_MIN + (float)ix * (GRID_X_MAX - GRID_X_MIN) / (GRID_X_STEPS - 1);
      
      for (int iy = 0; iy < GRID_Y_STEPS; ++iy) {
        float current_y = GRID_Y_MIN + (float)iy * (GRID_Y_MAX - GRID_Y_MIN) / (GRID_Y_STEPS - 1);
        
        for (int iz = 0; iz < GRID_Z_STEPS; ++iz) {
          float current_z = GRID_Z_MIN + (float)iz * (GRID_Z_MAX - GRID_Z_MIN) / (GRID_Z_STEPS - 1);
          
          // 6. Compute and normalize steering vector for this 3D grid point (spherical wave model)
          Complex steering_vector[NUM_MICS];
          float dx0 = current_x - mic_coords[0].x;
          float dy0 = current_y - mic_coords[0].y;
          float dz0 = current_z - mic_coords[0].z;
          float dist0 = sqrtf(dx0*dx0 + dy0*dy0 + dz0*dz0);
          
          // Calculate steering vector and its norm
          float norm_factor = 0.0f;
          for(int mic=0; mic<NUM_MICS; ++mic) {
              float dxi = current_x - mic_coords[mic].x;
              float dyi = current_y - mic_coords[mic].y;
              float dzi = current_z - mic_coords[mic].z;
              float disti = sqrtf(dxi*dxi + dyi*dyi + dzi*dzi);
              float time_diff = (disti - dist0) / SOUND_SPEED;
              float phase = 2.0f * PI * freq_hz * time_diff;
              
              // Store complex exponential
              arm_sin_cos_f32(phase, &steering_vector[mic].im, &steering_vector[mic].re);
              steering_vector[mic].im = -steering_vector[mic].im;  // exp(-j*theta)
              
              // Accumulate for normalization
              norm_factor += steering_vector[mic].re * steering_vector[mic].re + 
                            steering_vector[mic].im * steering_vector[mic].im;
          }
          
          // Normalize steering vector (add noise floor to avoid division by zero)
          norm_factor = sqrtf(norm_factor) + NOISE_FLOOR;
          for(int mic=0; mic<NUM_MICS; ++mic) {
              steering_vector[mic].re /= norm_factor;
              steering_vector[mic].im /= norm_factor;
          }

          // 7. Calculate pseudospectrum value with eigenvalue weighting
          float pseudo_spectrum = 0.0f;
          for (int n = 0; n < NUM_MICS - NUM_SOURCES; ++n) { // Project onto noise subspace
              // Skip very small eigenvalues
              if (eigenvalues[n] < MIN_EIGENVALUE) continue;
              
              Complex proj(0,0);
              for(int mic=0; mic<NUM_MICS; ++mic) {
                  proj = proj + steering_vector[mic].conj() * eigenvectors[mic][n];
              }
              // Weight by inverse of eigenvalue (MUSIC pseudospectrum)
              float weight = 1.0f / (eigenvalues[n] + NOISE_FLOOR);
              pseudo_spectrum += weight * (proj * proj.conj()).re;
          }
          // Add to the broadband spectrum
          if (pseudo_spectrum > 1e-9) {
              int idx = gridToIndex(ix, iy, iz);
              music_spectrum[idx] += pseudo_spectrum;
          }
        }
      }
    }
  }

  // 8. Find the peak in the 3D spectrum
  // Serial.println("8. Finding peak in spectrum...");
  float max_val = -1.0; 
  int max_ix = -1, max_iy = -1, max_iz = -1;
  
  for (int ix = 0; ix < GRID_X_STEPS; ix++) {
    for (int iy = 0; iy < GRID_Y_STEPS; iy++) {
      for (int iz = 0; iz < GRID_Z_STEPS; iz++) {
        int idx = gridToIndex(ix, iy, iz);
        if (music_spectrum[idx] > max_val) {
          max_val = music_spectrum[idx];
          max_ix = ix; 
          max_iy = iy;
          max_iz = iz;
        }
      }
    }
  }
  
  // 9. Convert peak index back to 3D coordinates
  // Serial.println("9. Converting peak to coordinates...");
  float est_x = GRID_X_MIN + (float)max_ix * (GRID_X_MAX - GRID_X_MIN) / (GRID_X_STEPS - 1);
  float est_y = GRID_Y_MIN + (float)max_iy * (GRID_Y_MAX - GRID_Y_MIN) / (GRID_Y_STEPS - 1);
  float est_z = GRID_Z_MIN + (float)max_iz * (GRID_Z_MAX - GRID_Z_MIN) / (GRID_Z_STEPS - 1);
  
  // Output the estimated 3D coordinates
  // if (Serial) {
  //   Serial.print("Estimated 3D Coords (mm): X=");
  //   Serial.print(est_x, 1);
  //   Serial.print(", Y=");
  //   Serial.print(est_y, 1);
  //   Serial.print(", Z=");
  //   Serial.println(est_z, 1);
  // }
  
  // Also output to Serial1 if connected (e.g., for Raspberry Pi)
  // if (Serial1) {
  //   Serial1.print("Estimated 3D Coords (mm): X=");
  //   Serial1.print(est_x, 1);
  //   Serial1.print(", Y=");
  //   Serial1.print(est_y, 1);
  //   Serial1.print(", Z=");
  //   Serial1.println(est_z, 1);
  // }
  
  // Detect and send sound sources
  send_sound_sources();
}
