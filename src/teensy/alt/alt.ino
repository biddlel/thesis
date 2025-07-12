#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>
#include <arm_math.h>
#include <arm_const_structs.h>
#include <complex>



// XSRP (SRP-PHAT) Parameters
const int FFT_SIZE = 256;                 // FFT size (power of 2)
const int NUM_MICS = 4;                   // Number of microphones
const float FS = AUDIO_SAMPLE_RATE_EXACT; // Sample rate (Hz)
const float C = 343.0f;                   // Speed of sound (m/s)
const float MIC_DISTANCE = 0.1f;          // Approximate distance between mics (m)
const int MAX_FREQ = 4000;                // Maximum frequency for XSRP (Hz)
const int MAX_FREQ_BIN = (int)(MAX_FREQ * FFT_SIZE / FS);

// Search grid parameters (in mm)
const float GRID_MIN_X = -1000.0f;
const float GRID_MAX_X = 1000.0f;
const float GRID_MIN_Y = -1000.0f;
const float GRID_MAX_Y = 1000.0f;
const float GRID_MIN_Z = 100.0f;
const float GRID_MAX_Z = 1000.0f;
const int GRID_STEPS_X = 20;
const int GRID_STEPS_Y = 20;
const int GRID_STEPS_Z = 5;


// Audio objects
AudioInputI2SQuad    i2s_quad;
// AudioFilterSPH0645   mic_correction[NUM_MICS];
AudioRecordQueue     queues[NUM_MICS];
AudioConnection*     patchCords[NUM_MICS];


// Microphone coordinates in mm (x, y, z)
const float mic_pos[NUM_MICS][3] = {
  { 96.41, 90.05, 0.0}, // Mic 1 (top-right)
  {-98.08, 89.86, 0.0}, // Mic 2 (top-left)
  {-99.13,-96.22, 0.0}, // Mic 3 (bottom-left)
  { 96.84,-96.78, 0.0 } // Mic 4 (bottom-right)
};

// FFT and buffer variables
arm_rfft_fast_instance_f32 fft_inst;
float fft_input[FFT_SIZE * 2];  // Real and imaginary parts interleaved
float fft_magnitude[FFT_SIZE / 2 + 1];
float fft_phase[FFT_SIZE / 2 + 1];

// Buffers for each microphone
float mic_buffers[NUM_MICS][FFT_SIZE];
std::complex<float> mic_fft[NUM_MICS][FFT_SIZE / 2 + 1];

// Window function (Hann)
float window[FFT_SIZE];

// Debug and timing variables
unsigned long last_debug_print = 0;
int blocks_collected = 0;

// Helper function to calculate distance between two 3D points
float distance(float x1, float y1, float z1, float x2, float y2, float z2) {
  float dx = x1 - x2;
  float dy = y1 - y2;
  float dz = z1 - z2;
  return sqrtf(dx*dx + dy*dy + dz*dz);
}

// Apply window function to data
void apply_window(float* data, int length) {
  for (int i = 0; i < length; i++) {
    data[i] *= window[i];
  }
}

// Function prototypes
void process_audio();
void compute_srp_phat();
float calculate_tdoa(float x, float y, float z, int mic1, int mic2);
void find_peak_position(float& x, float& y, float& z);
void compute_fft(float* input, std::complex<float>* output);

// Helper function to find peak in SRP map
void find_peak_position(float& x, float& y, float& z) {
  // This function can be enhanced with more sophisticated peak finding
  // For now, we'll just do a simple grid search in compute_srp_phat()
  x = 0.0f;
  y = 0.0f;
  z = 500.0f;  // Default to 500mm height if no peak found
}

// Helper function to calculate time difference of arrival (TDOA) in samples
float calculate_tdoa(float x, float y, float z, int mic1, int mic2) {
  // Calculate distance from source to each microphone (in mm)
  float dist1 = distance(x, y, z, mic_pos[mic1][0], mic_pos[mic1][1], mic_pos[mic1][2]);
  float dist2 = distance(x, y, z, mic_pos[mic2][0], mic_pos[mic2][1], mic_pos[mic2][2]);
  
  // Convert distance difference to time difference (in samples)
  float time_diff = (dist2 - dist1) / (C * 1000.0f) * FS; // C is in m/s, convert to mm/ms
  
  return time_diff;
}

// Function implementations
void compute_fft(float* input, std::complex<float>* output) {
  // Debug: Check input data
  for (int i = 0; i < FFT_SIZE; i++) {
    if (!isfinite(input[i])) {
      Serial.print("Invalid input to FFT at index ");
      Serial.println(i);
      return;
    }
  }
  
  // Copy input to real part, zero out imaginary part
  for (int i = 0; i < FFT_SIZE; i++) {
    fft_input[2*i] = input[i];
    fft_input[2*i+1] = 0;
  }
  
  // Compute FFT in place
  arm_rfft_fast_f32(&fft_inst, fft_input, fft_input, 0);
  
  // Copy to output complex array (only need first half + 1 for real FFT)
  for (int i = 0; i <= FFT_SIZE/2; i++) {
    output[i] = std::complex<float>(fft_input[2*i], fft_input[2*i+1]);
  }
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);  // For communication with Raspberry Pi
  while (!Serial && !Serial1 && millis() < 4000) {}
  
  Serial.println("Starting setup...");
  
  // Initialize audio memory and queues
  Serial.println("Initializing audio memory...");
  AudioMemory(100); // Increased from 20 to 100 for stability
  Serial.print("Audio memory initialized. Blocks: ");
  Serial.println(AudioMemoryUsageMax());
  
  // Enable the audio shield
  Serial.println("Enabling audio shield...");
  
  // Start the audio processing
  Serial.println("Starting audio processing...");
  AudioProcessorUsageMaxReset();
  AudioMemoryUsageMaxReset();

  Serial.println("Starting audio queues...");
  for(int i=0; i<NUM_MICS; ++i) {
    // correctionConnection[i] = new AudioConnection(i2s_quad, i, mic_correction[i], 0);
    patchCords[i] = new AudioConnection(i2s_quad, i, queues[i], 0);
    queues[i].begin();
    // Serial.print("Started queue for mic ");
    // Serial.println(i);
  }
  Serial.println("Audio queues started.");
  // Start the queues
     
  // Initialize FFT
  Serial.println("Initializing FFT...");
  if (arm_rfft_fast_init_f32(&fft_inst, FFT_SIZE) != ARM_MATH_SUCCESS) {
    Serial.println("FFT initialization failed!");
    while(1); // Halt on error
  }
  Serial.println("FFT initialized successfully.");
  
  // Initialize Hann window
  Serial.println("Initializing Hann window...");
  for (int i = 0; i < FFT_SIZE; i++) {
    window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (FFT_SIZE - 1)));
    if (!isfinite(window[i])) {
      Serial.print("Invalid window value at index ");
      Serial.println(i);
      while(1); // Halt on error
    }
  }
  Serial.println("Hann window initialized.");
  
  Serial.println("XSRP (SRP-PHAT) Sound Source Localization Started");
  Serial.print("Sample rate: ");
  Serial.print(FS);
  Serial.println(" Hz");
  Serial.print("FFT size: ");
  Serial.println(FFT_SIZE);
  Serial.print("Audio block samples: ");
  Serial.println(AUDIO_BLOCK_SAMPLES);
  Serial.print("FFT blocks needed: ");
  Serial.println(FFT_SIZE / AUDIO_BLOCK_SAMPLES);
  Serial.println("Setup complete. Starting main loop...");
}

void loop() {
  if (millis() - last_debug_print > 1000) {
    last_debug_print = millis();
    Serial.print("CPU: ");
    Serial.print(AudioProcessorUsage());
    Serial.print("% max: ");
    Serial.print(AudioProcessorUsageMax());
    Serial.print("%, Queues: ");
    for(int i = 0; i < NUM_MICS; i++) {
      Serial.print(queues[i].available());
      Serial.print("/");
    }
    Serial.println();
  }
  
  bool all_queues_ready = true;

  // Check if all queues have data available
  for(int i=0; i<NUM_MICS; ++i) {
    if(queues[i].available() == 0) { all_queues_ready = false; break; }
  }
  
  if (all_queues_ready) {
    Serial.println("All queues ready. Reading data...");
    
    for(int i=0; i<NUM_MICS; ++i) {
      int16_t *data = queues[i].readBuffer();
      if (data) {  // Check if data is valid
        // First copy the data to a temporary buffer
        int16_t temp_buffer[AUDIO_BLOCK_SAMPLES];
        memcpy(temp_buffer, data, AUDIO_BLOCK_SAMPLES * sizeof(int16_t));
    
        // Then convert to float and copy to mic_buffers
        for(int j = 0; j < AUDIO_BLOCK_SAMPLES; j++) {
            mic_buffers[i][(blocks_collected * AUDIO_BLOCK_SAMPLES) + j] = (float)temp_buffer[j] / 32768.0f;
        }

        queues[i].freeBuffer();
      }
    }
    
    blocks_collected++;
    
    // Check if we've collected enough blocks for processing
    if (blocks_collected >= (FFT_SIZE / AUDIO_BLOCK_SAMPLES)) {
      Serial.println("Data read. Processing audio...");
      process_audio();
      blocks_collected = 0;  // Reset for next batch
    }
  }
}

void process_audio() {
  Serial.println("Applying window...");
  // Apply window function to each microphone's buffer
  for (int m = 0; m < NUM_MICS; m++) {
    apply_window(mic_buffers[m], FFT_SIZE);
    
    // Debug: Check for NaN/Inf in windowed data
    for (int i = 0; i < FFT_SIZE; i++) {
      if (!isfinite(mic_buffers[m][i])) {
        Serial.print("Invalid value in mic ");
        Serial.print(m);
        Serial.print(" at index ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(mic_buffers[m][i], 6);
        return; // Skip this block if we find invalid data
      }
    }
  }
  
  Serial.println("Computing FFTs...");
  // Compute FFT for each microphone
  for (int m = 0; m < NUM_MICS; m++) {
    compute_fft(mic_buffers[m], mic_fft[m]);
  }
  
  Serial.println("Computing SRP-PHAT...");
  // Perform SRP-PHAT localization
  compute_srp_phat();
  Serial.println("SRP-PHAT finished.");
}

void compute_srp_phat() {
  float max_power = -1e12f;
  float best_x = 0, best_y = 0, best_z = 0;
  
  // Grid search over 3D space
  for (int ix = 0; ix <= GRID_STEPS_X; ix++) {
    float x = GRID_MIN_X + ix * (GRID_MAX_X - GRID_MIN_X) / GRID_STEPS_X;
    
    for (int iy = 0; iy <= GRID_STEPS_Y; iy++) {
      float y = GRID_MIN_Y + iy * (GRID_MAX_Y - GRID_MIN_Y) / GRID_STEPS_Y;
      
      for (int iz = 0; iz <= GRID_STEPS_Z; iz++) {
        float z = GRID_MIN_Z + iz * (GRID_MAX_Z - GRID_MIN_Z) / GRID_STEPS_Z;
        
        float power = 0.0f;
        
        // For each microphone pair, calculate GCC-PHAT
        for (int m1 = 0; m1 < NUM_MICS; m1++) {
          for (int m2 = m1 + 1; m2 < NUM_MICS; m2++) {
            // Calculate time difference of arrival (TDOA) in samples
            float tdoa = calculate_tdoa(x, y, z, m1, m2);
            
            std::complex<float> srp_sum(0.0f, 0.0f);

            // For each frequency bin, compute the GCC-PHAT contribution
            for (int k = 1; k < MAX_FREQ_BIN; k++) {  // Iterate up to MAX_FREQ_BIN
              // Cross-spectrum for the microphone pair
              std::complex<float> cross_spectrum = mic_fft[m1][k] * std::conj(mic_fft[m2][k]);
              
              // PHAT weighting (normalization)
              std::complex<float> gcc_phat = cross_spectrum / (std::abs(cross_spectrum) + 1e-10f);
              
              // Steering vector for this frequency bin and TDOA
              float expected_phase = 2.0f * PI * k * tdoa / FFT_SIZE;
              std::complex<float> steering_vector = std::exp(std::complex<float>(0, -expected_phase));
              
              srp_sum += gcc_phat * steering_vector;
            }
            power += srp_sum.real();
          }
        }
        
        // Update best position
        if (power > max_power) {
          max_power = power;
          best_x = x;
          best_y = y;
          best_z = z;
        }
      }
    }
  }

  // Output the estimated position
  Serial.print("X: ");
  Serial.print(best_x);
  Serial.print(" mm, Y: ");
  Serial.print(best_y);
  Serial.print(" mm, Z: ");
  Serial.print(best_z);
  Serial.print(" mm, Power: ");
  Serial.println(max_power);
  
  // Also send to Serial1 for Raspberry Pi
  if (Serial1.availableForWrite() > 0) {
    Serial1.print("X");
    Serial1.print(best_x);
    Serial1.print(",Y");
    Serial1.print(best_y);
    Serial1.print(",Z");
    Serial1.print(best_z);
    Serial1.println();
  }
}

/*
 * XSRP (eXtensible Steered Response Power) Implementation
 * 
 * This implementation uses the XSRP (eXtensible Steered Response Power) algorithm,
 * which is a variation of SRP-PHAT. It works by:
 * 1. Capturing audio from multiple microphones.
 * 2. Computing the FFT of each channel.
 * 3. For each point in a 3D grid, calculating the expected time delays (TDOA).
 * 4. For each microphone pair, computing the Generalized Cross-Correlation with 
 *    Phase Transform (GCC-PHAT).
 * 5. Applying a steering vector to the GCC-PHAT spectrum based on the TDOA.
 * 6. Summing the steered GCC-PHAT spectra over all microphone pairs and frequency bins
 *    up to a defined maximum frequency (MAX_FREQ) to get the SRP value.
 * 7. Finding the grid point with the maximum SRP value, which corresponds to the
 *    estimated sound source location.
 * 
 * This method is robust to noise and reverberation.
 */
