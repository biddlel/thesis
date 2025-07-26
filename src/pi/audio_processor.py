#!/usr/bin/env python3
"""
Audio Processor for Sound Source Localization with Quad I2S Support

This script:
1. Receives raw audio data from a Teensy 4.x via serial using I2S Quad mode
2. Parses the binary data format with error checking
3. Buffers the audio data for processing
4. Performs beamforming using the Acoular library
5. Visualizes sound source locations

Data Format (from Teensy):
- Header: 0x1F (1 byte) + timestamp (4 bytes, big-endian)
- Mean Values: 4 floats, 4 bytes each
- Footer: checksum (1 byte, XOR of all audio bytes) + 0x1E (1 byte)
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import serial
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
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
SAMPLE_RATE = 44100  # Still needed for timing

# Protocol constants
START_MARKER = 0x1F
END_MARKER = 0x1E
HEADER_SIZE = 5  # 1 byte start marker + 4 bytes timestamp
FOOTER_SIZE = 2  # 1 byte checksum + 1 byte end marker
MEAN_VALUES_SIZE = NUM_MICS * 4  # 4 bytes per float
CHUNK_SIZE = HEADER_SIZE + MEAN_VALUES_SIZE + FOOTER_SIZE

# Microphone positions in meters (matching Teensy order)
MIC_POSITIONS = {
    0: {'name': 'Top-Right',    'pos': [ 0.09641,  0.09005, 0.0]},
    1: {'name': 'Top-Left',     'pos': [-0.09808, 0.08986, 0.0]},
    2: {'name': 'Bottom-Left',  'pos': [-0.09913, -0.09622, 0.0]},
    3: {'name': 'Bottom-Right', 'pos': [0.09684, -0.09678, 0.0]}
}

@dataclass
class AudioPacket:
    """Container for mean values from the Teensy."""
    timestamp: int  # milliseconds since Teensy startup
    means: np.ndarray  # Shape: (NUM_MICS,)
    
    def __post_init__(self):
        assert self.means.shape == (NUM_MICS,), \
            f"Expected shape {(NUM_MICS,)}, got {self.means.shape}"

class TeensyAudioReceiver:
    """Handles receiving and processing mean values from the Teensy."""
    
    def __init__(self, port: str = SERIAL_PORT, baud_rate: int = BAUD_RATE):
        """Initialize the serial connection."""
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.packet_count = 0
        self.bytes_received = 0
        self.last_print_time = time.time()
        
    def connect(self) -> bool:
        """Connect to the Teensy via serial."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1.0
            )
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            print("Microphone order in stream:")
            for i in range(NUM_MICS):
                print(f"  {i}: {MIC_POSITIONS[i]['name']} at {MIC_POSITIONS[i]['pos']}")
            return True
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")
            return False
    
    def read_packet(self) -> Optional[AudioPacket]:
        """Read and parse a single line of mean values from the Teensy."""
        if self.serial_conn.in_waiting > 0:
            # Read a line from serial
            line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
            if not line:
                return None
                
            try:
                # Split the line into values
                values = line.split(',')
                if len(values) != NUM_MICS + 1:  # timestamp + NUM_MICS values
                    print(f"Unexpected number of values: {len(values)} (expected {NUM_MICS + 1})")
                    return None
                    
                timestamp = int(values[0])
                means = np.array([float(v) for v in values[1:]], dtype=np.float32)
                
                # Update statistics
                self.packet_count += 1
                self.bytes_received += len(line)
                
                # Print status periodically
                current_time = time.time()
                if current_time - self.last_print_time > 1.0:
                    rate = self.bytes_received / (current_time - self.last_print_time) / 1024
                    print(f"Received {self.packet_count} packets ({rate:.1f} KB/s)")
                    self.last_print_time = current_time
                    self.bytes_received = 0
                    
                return AudioPacket(timestamp, means)
                
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
                return None
        return None
    
    def process_in_background(self, callback):
        """
        Continuously read and process mean values in a background thread.
        
        Args:
            callback: Function to call with each received AudioPacket
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return
        
        print(f"Starting mean value processing")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                packet = self.read_packet()
                if packet:
                    callback(packet)
                time.sleep(0.001)  # Small sleep to prevent 100% CPU
                    
        except KeyboardInterrupt:
            print("\nStopping mean value processing")
        except Exception as e:
            print(f"Error in processing loop: {e}")
        finally:
            self.disconnect()

def setup_acoular_processing() -> Optional[Dict[str, Any]]:
    """
    Set up the Acoular processing pipeline for quad I2S input.
    
    Returns:
        A dictionary containing the Acoular processing context, or None on error.
    """
    try:
        import acoular
        from acoular import MicGeom, WNoiseGenerator, PointSource, FiltFiltOctave, \
                           PowerSpectra, RectGrid, BeamformerBase, BeamformerFunctional, \
                           RectGrid, SteeringVector, Environment
        
        print("Acoular version:", acoular.__version__)
        
        # Microphone geometry from MIC_POSITIONS
        mg = MicGeom()
        mic_positions = np.array([MIC_POSITIONS[i]['pos'] for i in range(NUM_MICS)]).T
        
        # Use the new pos_total attribute instead of mpos_tot
        if hasattr(mg, 'pos_total'):
            mg.pos_total = mic_positions
        else:
            # Fallback for older versions
            mg.mpos_tot = mic_positions  # type: ignore
        
        # Basic processing pipeline
        env = Environment(c=343.0)  # Speed of sound in m/s at 20°C
        grid = RectGrid(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z=1.0, increment=0.05)
        st = SteeringVector(grid=grid, mics=mg, env=env)
        
        print("Acoular processing pipeline initialized")
        print(f"Using {NUM_MICS} microphones")
        
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
    """Main function to demonstrate the mean value processing pipeline."""
    # Set up Acoular processing
    acoular_ctx = setup_acoular_processing()
    if not acoular_ctx:
        return
    
    # Set up the mean value receiver
    receiver = TeensyAudioReceiver()
    
    # Results queue for visualization
    result_queue = Queue()
    
    # Flag to control the processing thread
    processing = True
    
    # Create output directory if it doesn't exist
    output_dir = 'mean_values'
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for saved images
    image_counter = 0
    max_images = 100  # Maximum number of images to keep
    
    def process_audio_packet(packet: AudioPacket):
        # Put results in the queue for visualization
        result_queue.put({
            'means': packet.means,
            'timestamp': packet.timestamp
        })
    
    def update_plot():
        """Update and save the plot as an image file."""
        nonlocal image_counter
        
        try:
            # Get the latest result
            result = result_queue.get_nowait()
            
            # Create a new figure
            plt.figure(figsize=(10, 8))
            
            # Plot mean values
            plt.bar(range(NUM_MICS), result['means'])
            
            # Add title and labels
            plt.title(f'Mean Values at {result["timestamp"]} ms')
            plt.xlabel('Microphone Index')
            plt.ylabel('Mean Value')
            
            # Save the figure
            output_file = os.path.join(output_dir, f'mean_values_{image_counter % max_images:03d}.png')
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Saved mean values to {output_file}")
            image_counter += 1
            
            # Clean up old files if we've reached the maximum
            if image_counter >= max_images and image_counter % max_images == 0:
                old_file = os.path.join(output_dir, f'mean_values_{(image_counter - max_images) % max_images:03d}.png')
                if os.path.exists(old_file):
                    os.remove(old_file)
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            # Ensure we close any open figures to prevent memory leaks
            plt.close('all')
    
    try:
        # Start processing mean values in the background
        receiver.process_in_background(process_audio_packet)
        
        print("Processing mean values. Press Ctrl+C to stop...")
        
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
