#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

// Audio setup
AudioInputI2SQuad    i2s_quad;
AudioAnalyzePeak     peak1;
AudioAnalyzePeak     peak2;
AudioAnalyzePeak     peak3;
AudioAnalyzePeak     peak4;
AudioAmplifier       amp1;
AudioAmplifier       amp2;
AudioAmplifier       amp3;
AudioAmplifier       amp4;

// Audio connections
AudioConnection      patchCord1(i2s_quad, 0, amp1, 0);
AudioConnection      patchCord2(amp1, 0, peak1, 0);
AudioConnection      patchCord3(i2s_quad, 1, amp2, 0);
AudioConnection      patchCord4(amp2, 0, peak2, 0);
AudioConnection      patchCord5(i2s_quad, 2, amp3, 0);
AudioConnection      patchCord6(amp3, 0, peak3, 0);
AudioConnection      patchCord7(i2s_quad, 3, amp4, 0);
AudioConnection      patchCord8(amp4, 0, peak4, 0);

const float AMP_GAIN = 8.5;  // Amplifier gain

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) ;
  
  // Audio connections require memory
  AudioMemory(20);
  
  // Set amplifier gains
  amp1.gain(AMP_GAIN);
  amp2.gain(AMP_GAIN);
  amp3.gain(AMP_GAIN);
  amp4.gain(AMP_GAIN);
  
  Serial.println("Microphone Level Monitor (Gain: 8.5x)");
  Serial.println("------------------------------------");
  Serial.println("Mic1 (TR)  Mic2 (TL)  Mic3 (BL)  Mic4 (BR)");
  Serial.println("------------------------------------");
  
  // Wait for user input to start
  while (!Serial.available()) {
    delay(10);
  }
  while (Serial.available()) Serial.read(); // Clear the input buffer
}

void loop() {
  // Print values in a single line
  Serial.println("Mic1 (TR)  Mic2 (TL)  Mic3 (BL)  Mic4 (BR)");
  Serial.print(peak1.read() * 100, 2);
  Serial.print("  ");
  Serial.print(peak2.read() * 100, 2);
  Serial.print("  ");
  Serial.print(peak3.read() * 100, 2);
  Serial.print("  ");
  Serial.print(peak4.read() * 100, 2);
  Serial.println();
  
  // Small delay to prevent flooding the serial port
  delay(1000);
}