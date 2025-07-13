import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless operation
import matplotlib.pyplot as plt
import serial
import struct
import os
import time
from datetime import datetime

# Configuration
SERIAL_PORT = '/dev/ttyACM0'  # Update this to your serial port
BAUD_RATE = 115200
OUTPUT_DIR = 'spectrum_plots'
LOG_FILE = 'spectrum_log.txt'
PLOT_INTERVAL = 1.0  # Minimum seconds between plots
MAX_PLOTS = 1000

# Structure for the header (must match the C++ struct)
class SpectrumHeader:
    def __init__(self, data):
        (self.magic,
         self.x_steps, self.y_steps, self.z_steps,
         self.x_min, self.x_max,
         self.y_min, self.y_max,
         self.z_min, self.z_max,
         self.data_size,
         self.checksum) = struct.unpack('<IIIIfffffIII', data)
    
    def calculate_checksum(self, data):
        return sum(data[:40])  # Sum of first 40 bytes (before checksum)

def log_message(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')

def read_spectrum(ser):
    """Read a complete spectrum data packet"""
    try:
        # Read header (44 bytes: 4 + 3*4 + 6*4 + 4 + 4)
        header_data = ser.read(44)
        if len(header_data) != 44:
            return None
            
        header = SpectrumHeader(header_data)
        
        # Verify magic number
        if header.magic != 0x4D555349:  # 'MUSI'
            log_message(f"Invalid magic number: {hex(header.magic)}")
            return None
            
        # Verify checksum
        if header.calculate_checksum(header_data) != header.checksum:
            log_message("Checksum verification failed")
            return None
        
        # Calculate total data size
        total_bytes = header.data_size * 4  # 4 bytes per float
        data = bytearray()
        
        # Read data in chunks
        while len(data) < total_bytes:
            chunk = ser.read(min(1024, total_bytes - len(data)))
            if not chunk:
                log_message("Incomplete data")
                return None
            data.extend(chunk)
        
        # Read end marker
        end_marker = ser.read(4)
        if len(end_marker) != 4 or end_marker != b'\xFF\xFF\xFF\xFF':
            log_message("Invalid end marker")
            return None
        
        # Convert to numpy array
        spectrum = np.frombuffer(data, dtype=np.float32)
        
        # Reshape to 3D grid
        spectrum = spectrum.reshape((header.x_steps, header.y_steps, header.z_steps))
        
        return {
            'spectrum': spectrum,
            'x': np.linspace(header.x_min, header.x_max, header.x_steps),
            'y': np.linspace(header.y_min, header.y_max, header.y_steps),
            'z': np.linspace(header.z_min, header.z_max, header.z_steps),
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        log_message(f"Error reading spectrum: {e}")
        return None

def plot_spectrum(spectrum_data, output_dir, plot_number):
    """Create and save a visualization of the spectrum"""
    try:
        # Find slice with maximum power
        max_z_idx = np.argmax(np.max(spectrum_data['spectrum'], axis=(0, 1)))
        z_value = spectrum_data['z'][max_z_idx]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot XY slice at max Z
        plt.imshow(spectrum_data['spectrum'][:, :, max_z_idx].T,
                  extent=[spectrum_data['x'][0], spectrum_data['x'][-1],
                          spectrum_data['y'][0], spectrum_data['y'][-1]],
                  origin='lower', aspect='auto', cmap='viridis')
        
        plt.colorbar(label='Power (dB)')
        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.title(f'Sound Source Localization (Z = {z_value:.1f} mm)')
        
        # Add timestamp
        timestamp = spectrum_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        plt.figtext(0.5, 0.01, f'Last Update: {timestamp}', 
                   ha='center', fontsize=8, style='italic')
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'spectrum_{plot_number:04d}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Create/update symlink to latest plot
        latest_path = os.path.join(output_dir, 'latest.png')
        if os.path.lexists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(output_path), latest_path)
        
        return output_path
        
    except Exception as e:
        log_message(f"Error creating plot: {e}")
        return None

def cleanup_old_plots(output_dir, max_plots):
    """Remove old plot files to save disk space"""
    try:
        if not os.path.exists(output_dir):
            return
            
        # Get all plot files
        plot_files = [f for f in os.listdir(output_dir) 
                     if f.startswith('spectrum_') and f.endswith('.png')]
        plot_files.sort()
        
        # Remove oldest files if we have too many
        while len(plot_files) > max_plots:
            oldest = plot_files.pop(0)
            os.remove(os.path.join(output_dir, oldest))
            log_message(f"Removed old plot: {oldest}")
            
    except Exception as e:
        log_message(f"Error cleaning up old plots: {e}")

def main():
    log_message("Starting spectrum visualizer")
    log_message(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    log_message("Press Ctrl+C to stop")
    
    try:
        # Open serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
        log_message(f"Connected to {ser.name}")
        
        plot_number = 0
        last_plot_time = 0
        last_log_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Read data from serial
            spectrum_data = read_spectrum(ser)
            
            # If we have valid data and enough time has passed, create a plot
            if spectrum_data and (current_time - last_plot_time) >= PLOT_INTERVAL:
                plot_path = plot_spectrum(spectrum_data, OUTPUT_DIR, plot_number)
                if plot_path:
                    log_message(f"Saved plot: {plot_path}")
                    
                    # Log peak position
                    max_idx = np.unravel_index(np.argmax(spectrum_data['spectrum']), 
                                             spectrum_data['spectrum'].shape)
                    x_pos = spectrum_data['x'][max_idx[0]]
                    y_pos = spectrum_data['y'][max_idx[1]]
                    z_pos = spectrum_data['z'][max_idx[2]]
                    log_message(f"Peak at: X={x_pos:.1f}mm, Y={y_pos:.1f}mm, Z={z_pos:.1f}mm")
                
                # Clean up old plots periodically
                if plot_number % 10 == 0:
                    cleanup_old_plots(OUTPUT_DIR, MAX_PLOTS)
                
                plot_number += 1
                last_plot_time = current_time
            
            # Periodic status update
            if current_time - last_log_time > 60:  # Every minute
                log_message(f"Still running... {plot_number} plots saved")
                last_log_time = current_time
                
    except serial.SerialException as e:
        log_message(f"Serial port error: {e}")
    except KeyboardInterrupt:
        log_message("Stopped by user")
    except Exception as e:
        log_message(f"Unexpected error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            log_message("Serial port closed")
        log_message("Spectrum visualizer stopped")

if __name__ == "__main__":
    main()