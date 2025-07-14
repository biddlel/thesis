#include <Audio.h>
#include <arm_math.h>
#include "linalg.h"
#include "AudioFilterSPH0645.h"
#include <ArduinoJson.h>

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


struct SpectrumHeader {
  uint32_t magic;         // 'MUSI' as 32-bit value
  uint32_t x_steps;       // Number of steps in X dimension
  uint32_t y_steps;       // Number of steps in Y dimension
  uint32_t z_steps;       // Number of steps in Z dimension
  float x_min, x_max;     // X dimension bounds
  float y_min, y_max;     // Y dimension bounds
  float z_min, z_max;     // Z dimension bounds
  uint32_t data_size;     // Total number of floats in the spectrum data
  uint32_t checksum;      // Simple checksum of the header
};

// Calculate a simple checksum for the header
uint32_t calculate_checksum(const uint8_t* data, size_t length) {
  uint32_t sum = 0;
  for (size_t i = 0; i < length; i++) {
      sum += data[i];
  }
  return sum;
}

// Binary message header
struct MessageHeader {
  uint32_t magic;      // Magic number to identify the start of a message
  uint32_t length;     // Length of the JSON data
  uint32_t checksum;   // Simple checksum of the JSON data
};

bool send_spectrum() {
  // Create a JSON document
  JsonDocument doc;
  
  // Add metadata
  doc["x_steps"] = GRID_X_STEPS;
  doc["y_steps"] = GRID_Y_STEPS;
  doc["z_steps"] = GRID_Z_STEPS;
  doc["x_min"] = GRID_X_MIN;
  doc["x_max"] = GRID_X_MAX;
  doc["y_min"] = GRID_Y_MIN;
  doc["y_max"] = GRID_Y_MAX;
  doc["z_min"] = GRID_Z_MIN;
  doc["z_max"] = GRID_Z_MAX;
  
  // Add spectrum data as a flat array
  JsonArray spectrum_data = doc.createNestedArray("spectrum_data");
  int total_points = GRID_X_STEPS * GRID_Y_STEPS * GRID_Z_STEPS;
  
  // Copy data to JSON array (flatten the 3D array)
  for (int i = 0; i < total_points; i++) {
    int x = i % GRID_X_STEPS;
    int y = (i / GRID_X_STEPS) % GRID_Y_STEPS;
    int z = i / (GRID_X_STEPS * GRID_Y_STEPS);
    
    // Convert 3D index to 1D index in the music_spectrum array
    int idx = x + GRID_X_STEPS * (y + GRID_Y_STEPS * z);
    spectrum_data.add(music_spectrum[idx]);
  }
  
  // Serialize JSON to a string
  String json_string;
  serializeJson(doc, json_string);
  
  // Calculate checksum
  uint32_t checksum = 0;
  for (size_t i = 0; i < json_string.length(); i++) {
    checksum += (uint8_t)json_string[i];
  }
  
  // Prepare message header
  MessageHeader header;
  header.magic = 0x4A534F4E;  // 'JSON' in ASCII
  header.length = json_string.length();
  header.checksum = checksum;
  
  // Send header
  Serial1.write((const uint8_t*)&header, sizeof(header));
  
  // Send JSON data
  size_t bytes_written = 0;
  const char* json_data = json_string.c_str();
  
  // Write in chunks to avoid blocking
  while (bytes_written < header.length) {
    size_t chunk_size = min(64, (int)(header.length - bytes_written));
    size_t written = Serial1.write(json_data + bytes_written, chunk_size);
    if (written == 0) {
      // Write failed
      return false;
    }
    bytes_written += written;
  }
  
  // Send a newline for good measure (optional)
  Serial1.write('\n');
  
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
  if (Serial) {
    Serial.print("Estimated 3D Coords (mm): X=");
    Serial.print(est_x, 1);
    Serial.print(", Y=");
    Serial.print(est_y, 1);
    Serial.print(", Z=");
    Serial.println(est_z, 1);
  }
  
  // Also output to Serial1 if connected (e.g., for Raspberry Pi)
  // if (Serial1) {
  //   Serial1.print("Estimated 3D Coords (mm): X=");
  //   Serial1.print(est_x, 1);
  //   Serial1.print(", Y=");
  //   Serial1.print(est_y, 1);
  //   Serial1.print(", Z=");
  //   Serial1.println(est_z, 1);
  // }
  
  // Send the complete spectrum data
  send_spectrum();
}
