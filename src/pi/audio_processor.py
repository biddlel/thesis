#!/usr/bin/env python3
"""
Audio Processor for Sound Source Localization

This script:
1. Reads raw audio data from a Teensy via serial
2. Parses the binary data format
3. Buffers the audio data
4. Prepares it for processing with the Acoular library
"""

import serial
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import struct
import threading
from queue import Queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy.signal import butter, lfilter

# Configuration
SERIAL_PORT = '/dev/ttyAMA0'  # Update this to match your serial port
BAUD_RATE = 115200
NUM_MICS = 4
BUFFER_SIZE = 1024  # Must match the Teensy BUFFER_SIZE
SAMPLE_RATE = 44100  # Must match the Teensy SAMPLE_RATE

# Protocol constants
START_MARKER = 0x1F
END_MARKER = 0x1E
HEADER_SIZE = 5  # 1 byte start marker + 4 bytes timestamp
FOOTER_SIZE = 2  # 1 byte checksum + 1 byte end marker
SAMPLE_BYTES = 2  # 16-bit samples
CHUNK_SIZE = NUM_MICS * BUFFER_SIZE * SAMPLE_BYTES + HEADER_SIZE + FOOTER_SIZE

@dataclass
class AudioPacket:
    """Container for a single packet of audio data from the Teensy."""
    timestamp: int  # milliseconds since Teensy startup
    data: np.ndarray  # Shape: (NUM_MICS, BUFFER_SIZE)
    
    def __post_init__(self):
        assert self.data.shape == (NUM_MICS, BUFFER_SIZE), \
            f"Expected shape {(NUM_MICS, BUFFER_SIZE)}, got {self.data.shape}"

class TeensyAudioReceiver:
    """Handles receiving and processing audio data from the Teensy."""
    
    def __init__(self, port: str = SERIAL_PORT, baud_rate: int = BAUD_RATE):
        """Initialize the serial connection and buffers."""
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.buffer = bytearray()
        self.packet_count = 0
        self.bytes_received = 0
        self.last_print_time = time.time()
        
        # Circular buffer for storing audio data (keeps last N packets)
        self.audio_buffer = deque(maxlen=10)  # Adjust maxlen as needed
        
    def connect(self) -> bool:
        """Connect to the Teensy via serial."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1.0
            )
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from serial port")
    
    def read_packet(self) -> Optional[AudioPacket]:
        """
        Read and parse a single audio packet from the serial connection.
        
        Returns:
            AudioPacket if a complete packet was received, None otherwise.
        """
        # Read any available data
        if self.serial_conn.in_waiting > 0:
            self.buffer.extend(self.serial_conn.read(self.serial_conn.in_waiting))
        
        # Look for start marker
        while len(self.buffer) >= 1:
            if self.buffer[0] != START_MARKER:
                # Discard bytes until we find a start marker
                self.buffer.pop(0)
                continue
                
            # Check if we have a complete packet
            if len(self.buffer) < CHUNK_SIZE:
                return None  # Incomplete packet
                
            # Extract the packet
            packet = self.buffer[:CHUNK_SIZE]
            self.buffer = self.buffer[CHUNK_SIZE:]
            
            # Verify end marker
            if packet[-1] != END_MARKER:
                print("Error: Invalid end marker")
                return None
                
            # Parse the packet
            try:
                # Extract timestamp (big-endian uint32)
                timestamp = int.from_bytes(packet[1:5], byteorder='big')
                
                # Extract audio data (4 channels interleaved)
                audio_data = np.frombuffer(
                    packet[HEADER_SIZE:-FOOTER_SIZE], 
                    dtype=np.int16
                ).reshape(NUM_MICS, BUFFER_SIZE)
                
                # Verify checksum
                checksum = packet[-2]  # Second to last byte
                calculated_checksum = 0
                for sample in audio_data.flat:
                    calculated_checksum ^= (sample & 0xFF)
                    calculated_checksum ^= ((sample >> 8) & 0xFF)
                
                if checksum != calculated_checksum:
                    print(f"Checksum error: expected {checksum:02X}, got {calculated_checksum:02X}")
                    return None
                
                self.packet_count += 1
                self.bytes_received += CHUNK_SIZE
                
                # Print status periodically
                current_time = time.time()
                if current_time - self.last_print_time > 1.0:
                    print(f"Received {self.packet_count} packets ({self.bytes_received/1024:.1f} KB)")
                    self.last_print_time = current_time
                
                return AudioPacket(timestamp, audio_data.astype(np.float32) / 32768.0)  # Convert to float32 [-1, 1]
                
            except Exception as e:
                print(f"Error parsing packet: {e}")
                return None
        
        return None
    
    def process_in_background(self, callback):
        """
        Continuously read and process audio data in a background thread.
        
        Args:
            callback: Function to call with each received AudioPacket
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return
        
        print(f"Starting audio processing at {SAMPLE_RATE} Hz")
        
        try:
            while True:
                packet = self.read_packet()
                if packet:
                    self.audio_buffer.append(packet)
                    if callback:
                        callback(packet)
                time.sleep(0.001)  # Small sleep to prevent 100% CPU usage
                
        except KeyboardInterrupt:
            print("\nStopping audio processing")
        except Exception as e:
            print(f"Error in processing loop: {e}")
        finally:
            self.disconnect()

def setup_acoular_processing():
    """
    Set up the Acoular processing pipeline.
    
    Returns:
        A configured Acoular processing pipeline.
    """
    try:
        import acoular
        from acoular import MicGeom, WNoiseGenerator, PointSource, FiltFiltOctave, \
                           PowerSpectra, RectGrid, BeamformerBase, BeamformerFunctional, \
                           RectGrid, SteeringVector, Environment
        
        print("Acoular version:", acoular.__version__)
        
        # Microphone geometry (update with your actual mic positions in meters)
        # Positions should match the order in the Teensy code (Top-Right, Top-Left, Bottom-Left, Bottom-Right)
        mg = MicGeom()
        mg.mpos_tot = np.array([
            [0.09641,  0.09005, 0.0],  # Top-Right
            [-0.09808, 0.08986, 0.0],  # Top-Left
            [-0.09913, -0.09622, 0.0], # Bottom-Left
            [0.09684, -0.09678, 0.0]   # Bottom-Right
        ]).T  # Transpose to match Acoular's expected shape (3, n_mics)
        
        # Basic processing pipeline
        env = Environment(c=343.0)  # Speed of sound in m/s
        grid = RectGrid(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z=1.0, increment=0.05)
        st = SteeringVector(grid=grid, mics=mg, env=env)
        
        print("Acoular processing pipeline initialized")
        return {
            'mic_geometry': mg,
            'environment': env,
            'grid': grid,
            'steering_vector': st
        }
        
    except ImportError as e:
        print("Error importing Acoular. Make sure it's installed:")
        print("pip install acoular")
        print(f"Import error: {e}")
        return None


def main():
    """Main function to demonstrate the audio processing pipeline."""
    # Set up Acoular processing
    acoular_ctx = setup_acoular_processing()
    if not acoular_ctx:
        return
    
    # Set up the audio receiver
    receiver = TeensyAudioReceiver()
    
    # Audio buffer for 1 second of data
    buffer_duration = 1.0  # seconds
    samples_needed = int(SAMPLE_RATE * buffer_duration)
    audio_buffer = np.zeros((NUM_MICS, samples_needed), dtype=np.float32)
    buffer_pos = 0
    
    # Results queue for visualization
    result_queue = Queue()
    
    # Lock for thread-safe buffer access
    buffer_lock = threading.Lock()
    
    # Flag to control the processing thread
    processing = True
    
    # Create output directory if it doesn't exist
    output_dir = 'sound_maps'
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for saved images
    image_counter = 0
    max_images = 100  # Maximum number of images to keep
    
    def process_audio_packet(packet: AudioPacket):
        nonlocal buffer_pos
        
        with buffer_lock:
            # Calculate how many samples we can add to the buffer
            remaining_space = samples_needed - buffer_pos
            add_samples = min(packet.data.shape[1], remaining_space)
            
            if add_samples > 0:
                # Add the new samples to the buffer
                audio_buffer[:, buffer_pos:buffer_pos + add_samples] = \
                    packet.data[:, :add_samples]
                buffer_pos += add_samples
                
                # If buffer is full, process it
                if buffer_pos >= samples_needed:
                    # Make a copy of the buffer for processing
                    process_buffer = audio_buffer.copy()
                    buffer_pos = 0  # Reset buffer position
                    
                    # Start a new thread for beamforming to avoid blocking
                    threading.Thread(
                        target=run_beamforming,
                        args=(process_buffer, result_queue, acoular_ctx)
                    ).start()
    
    def run_beamforming(audio_data, result_queue, ac_ctx):
        try:
            # Create a TimeSamples object with our data
            from acoular import TimeSamples, MicGeom, RectGrid, \
                               SteeringVector, BeamformerBase, \
                               BeamformerFunctional, FiltFiltOctave, \
                               PowerSpectra, Environment, L_p
            
            # Create a TimeSamples-like object with our data
            class AudioBuffer(TimeSamples):
                def __init__(self, data, sample_rate):
                    self.data = data
                    self.sample_rate = sample_rate
                    self.numsamples_fh = data.shape[1]
                    self.numchannels = data.shape[0]
                
                def data_upto(self, samples):
                    return self.data[:, :samples]
                
                def _get_data(self, num=0):
                    return self.data
            
            # Create time data object
            ts = AudioBuffer(audio_data, SAMPLE_RATE)
            
            # Configure processing chain
            ps = PowerSpectra(
                time_data=ts,
                block_size=128,  # FFT size
                window='Hanning'
            )
            
            # Create a grid for the beamforming map
            grid = ac_ctx['grid']
            
            # Configure beamformer (using functional beamformer as in the example)
            bb = BeamformerFunctional(
                freq_data=ps,
                grid=grid,
                steer=ac_ctx['steering_vector'],
                r_diag=True,
                # Functional beamformer parameters
                gamma=3,  # Controls the dynamic range
                dyn_range=40.0  # Dynamic range in dB
            )
            
            # Calculate the beamforming map
            pm = bb.synthetic(8000, 1)  # 8 kHz, single frequency band
            
            # Get the maximum value position
            max_idx = np.argmax(pm)
            max_pos = grid.pos.T[max_idx]
            
            # Put results in the queue for visualization
            result_queue.put({
                'map': pm.reshape(grid.shape),
                'max_pos': max_pos,
                'timestamp': time.time()
            })
            
        except Exception as e:
            print(f"Error in beamforming: {e}")
    
    def update_plot():
        """Update and save the plot as an image file."""
        nonlocal image_counter
        
        try:
            # Get the latest result
            result = result_queue.get_nowait()
            
            # Create a new figure
            plt.figure(figsize=(10, 8))
            
            # Create a heatmap of the beamforming result
            plt.imshow(
                result['map'],
                extent=[
                    acoular_ctx['grid'].x_min,
                    acoular_ctx['grid'].x_max,
                    acoular_ctx['grid'].y_min,
                    acoular_ctx['grid'].y_max
                ],
                origin='lower',
                aspect='auto',
                cmap='hot',
                norm=mcolors.Normalize(vmin=0, vmax=1)
            )
            
            # Add colorbar
            plt.colorbar(label='Relative Power (dB)')
            
            # Mark the maximum position
            plt.scatter(
                result['max_pos'][0],
                result['max_pos'][1],
                c='cyan',
                marker='x',
                s=100,
                linewidths=2
            )
            
            # Add title and labels
            plt.title(f'Sound Source Localization\nMax at: ({result["max_pos"][0]:.2f}, {result["max_pos"][1]:.2f}) m')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            
            # Plot microphone positions
            plt.scatter(
                acoular_ctx['mic_geometry'].mpos[0],
                acoular_ctx['mic_geometry'].mpos[1],
                c='blue',
                marker='o',
                label='Microphones'
            )
            
            plt.legend()
            plt.tight_layout()
            
            # Save the figure
            output_file = os.path.join(output_dir, f'sound_map_{image_counter % max_images:03d}.png')
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Saved sound map to {output_file}")
            image_counter += 1
            
            # Clean up old files if we've reached the maximum
            if image_counter >= max_images and image_counter % max_images == 0:
                old_file = os.path.join(output_dir, f'sound_map_{(image_counter - max_images) % max_images:03d}.png')
                if os.path.exists(old_file):
                    os.remove(old_file)
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            # Ensure we close any open figures to prevent memory leaks
            plt.close('all')
    
    try:
        # Start processing audio in the background
        receiver.process_in_background(process_audio_packet)
        
        print("Processing audio. Press Ctrl+C to stop...")
        
        # Main loop
        while processing:
            try:
                # Process any results in the queue
                if not result_queue.empty():
                    update_plot()
                # Small sleep to prevent high CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
                
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        processing = False
        receiver.disconnect()
        print("Processing stopped.")


if __name__ == "__main__":
    main()
