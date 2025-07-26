#include <Audio.h>
#include <Wire.h>
#include <SPI.h>

// Audio setup
AudioInputI2SQuad        i2s1;
AudioRecordQueue         queue1;
AudioRecordQueue         queue2;
AudioRecordQueue         queue3;
AudioRecordQueue         queue4;

// Audio connections - Each I2S has 2 channels (left and right)
// Using separate I2S interfaces for better channel separation
AudioConnection          patchCord1(i2s1, 0, queue1, 0);  // Mic 1 (left channel of i2s1)
AudioConnection          patchCord2(i2s1, 1, queue2, 0);  // Mic 2 (right channel of i2s1)
AudioConnection          patchCord3(i2s1, 2, queue3, 0);  // Mic 3 (left channel of i2s2)
AudioConnection          patchCord4(i2s1, 3, queue4, 0);  // Mic 4 (right channel of i2s2)

// Microphone array configuration
const int NUM_MICS = 4;                     // Number of microphones
const int SAMPLE_RATE = 44100;              // Audio sample rate (Hz)
const int SAMPLES_PER_BLOCK = 128;          // Audio block size
const int CHANNELS_PER_MIC = 1;             // I2S has 1 channel per mic

// Microphone positions (in mm)
struct MicCoordinate { 
    float x, y, z; 
    const char* name;
};

MicCoordinate mic_coords[NUM_MICS] = {
    { 96.41,  90.05, 0.0, "Top-Right"},    // Mic 1
    {-98.08,  89.86, 0.0, "Top-Left"},     // Mic 2
    {-99.13, -96.22, 0.0, "Bottom-Left"},  // Mic 3
    { 96.84, -96.78, 0.0, "Bottom-Right"}  // Mic 4
};

// Audio buffer
const int BUFFER_SIZE = 1024;  // Number of samples per channel per transmission
int16_t audio_buffer[NUM_MICS][BUFFER_SIZE];
volatile int sample_count = 0;

// Serial communication
const byte START_MARKER = 0x1F;
const byte END_MARKER = 0x1E;

// Function prototypes
void send_audio_buffer();
void print_mic_positions();

void setup() {
    // Initialize serial communication
    Serial.begin(115200);  // USB Serial for debugging
    Serial1.begin(115200); // Hardware serial for communication with Pi
    
    // Wait for serial port to connect (up to 4 seconds)
    while (!Serial && millis() < 4000) {}
    
    // Allocate audio memory (increase if needed for 4 channels)
    AudioMemory(60);
    
    // Initialize audio queues
    queue1.begin();
    queue2.begin();
    queue3.begin();
    queue4.begin();
    
    // Print configuration
    Serial.println("\n=== Teensy Audio Streamer ===");
    Serial.println("Streaming raw audio from 4 microphones");
    Serial.println("----------------------------");
    
    // Print microphone positions
    print_mic_positions();
    
    Serial.print("Sample rate: ");
    Serial.print(SAMPLE_RATE);
    Serial.println(" Hz");
    Serial.print("Samples per block: ");
    Serial.println(SAMPLES_PER_BLOCK);
    Serial.println("Ready!\n");
}

void loop() {
    // Check if we have audio data available from all queues
    if (queue1.available() >= 1 && 
        queue2.available() >= 1 && 
        queue3.available() >= 1 && 
        queue4.available() >= 1) {
        
        // Get audio blocks from each queue
        int16_t *block1 = queue1.readBuffer();
        int16_t *block2 = queue2.readBuffer();
        int16_t *block3 = queue3.readBuffer();
        int16_t *block4 = queue4.readBuffer();
        
        // Copy samples to our buffer
        for (int i = 0; i < SAMPLES_PER_BLOCK; i++) {
            if (sample_count < BUFFER_SIZE) {
                // Each queue provides one channel of audio data
                audio_buffer[0][sample_count] = block1[i];  // Mic 1
                audio_buffer[1][sample_count] = block2[i];  // Mic 2
                audio_buffer[2][sample_count] = block3[i];  // Mic 3
                audio_buffer[3][sample_count] = block4[i];  // Mic 4
                
                sample_count++;
                
                // If buffer is full, send it and reset
                if (sample_count >= BUFFER_SIZE) {
                    send_audio_buffer();
                    sample_count = 0;
                }
            }
        }
        
        // Free the audio blocks
        queue1.freeBuffer();
        queue2.freeBuffer();
        queue3.freeBuffer();
        queue4.freeBuffer();
    }
}

// Print microphone positions for verification
void print_mic_positions() {
    Serial.println("Microphone Positions (mm):");
    for (int i = 0; i < NUM_MICS; i++) {
        Serial.print("  Mic ");
        Serial.print(i + 1);
        Serial.print(" (");
        Serial.print(mic_coords[i].name);
        Serial.print("): ");
        Serial.print("X=");
        Serial.print(mic_coords[i].x, 2);
        Serial.print(", Y=");
        Serial.print(mic_coords[i].y, 2);
        Serial.print(", Z=");
        Serial.print(mic_coords[i].z, 2);
        Serial.println();
    }
    Serial.println();
}

// Send audio buffer over serial
void send_audio_buffer() {
    // Send start marker
    Serial1.write(START_MARKER);
    
    // Send timestamp (milliseconds since startup)
    uint32_t timestamp = millis();
    Serial1.write((uint8_t*)&timestamp, 4);
    
    // Send audio data for all channels
    for (int ch = 0; ch < NUM_MICS; ch++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            int16_t sample = audio_buffer[ch][i];
            Serial1.write((uint8_t*)&sample, 2);
        }
    }
    
    // Calculate and send checksum (simple XOR)
    uint8_t checksum = 0;
    for (int ch = 0; ch < NUM_MICS; ch++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            checksum ^= (audio_buffer[ch][i] & 0xFF);
            checksum ^= ((audio_buffer[ch][i] >> 8) & 0xFF);
        }
    }
    Serial1.write(checksum);
    
    // Send end marker
    Serial1.write(END_MARKER);
    
    // Debug output
    static unsigned long last_print = 0;
    if (millis() - last_print > 1000) {
        Serial.print("Sent ");
        Serial.print(BUFFER_SIZE);
        Serial.println(" samples per channel");
        last_print = millis();
    }
}
