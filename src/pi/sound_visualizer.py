import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless operation
import matplotlib.pyplot as plt
import serial
import json
import os
import time
from datetime import datetime

# Configuration
SERIAL_PORT = '/dev/ttyAMA0'  # Update this to your serial port
BAUD_RATE = 115200
OUTPUT_DIR = 'spectrum_plots'
LOG_FILE = 'spectrum_log.txt'
PLOT_INTERVAL = 1.0  # Minimum seconds between plots
MAX_PLOTS = 1000

# JSON message separator
MESSAGE_SEPARATOR = b'\n\n'  # Double newline to separate JSON messages

# Track if we've logged before in this session
_first_log = True

def log_message(message):
    """Log messages with timestamp"""
    global _first_log
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    # Clear the log file on first use, then append for subsequent messages
    mode = 'w' if _first_log else 'a'
    with open(LOG_FILE, mode) as f:
        f.write(log_entry + '\n')
    _first_log = False

def dump_hex(data, bytes_per_line=16):
    """Convert binary data to a hex dump string"""
    result = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        result.append(f"{i:04x}: {hex_str.ljust(bytes_per_line*3)}  {ascii_str}")
    return '\n'.join(result)

def find_magic(ser):
    """Legacy function kept for compatibility - not used in JSON mode"""
    log_message("Note: find_magic() is not used in JSON mode")
    return False, None

def read_json_message(ser, timeout=5.0):
    """Read a complete JSON message from serial"""
    start_time = time.time()
    buffer = bytearray()
    
    while True:
        if time.time() - start_time > timeout:
            log_message("Timeout waiting for message")
            return None
            
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            buffer += chunk
            
            # Look for message separator
            if MESSAGE_SEPARATOR in buffer:
                message_end = buffer.find(MESSAGE_SEPARATOR)
                message = buffer[:message_end].decode('utf-8', errors='ignore')
                buffer = buffer[message_end + len(MESSAGE_SEPARATOR):]
                
                try:
                    data = json.loads(message)
                    log_message("Successfully parsed JSON message")
                    return data
                except json.JSONDecodeError as e:
                    log_message(f"JSON decode error: {e}")
                    log_message(f"Message content: {message[:200]}...")  # Log first 200 chars
                    continue
        else:
            time.sleep(0.01)  # Small delay to prevent busy waiting

def read_spectrum(ser):
    """Read a complete spectrum data packet in JSON format"""
    try:
        # Read a complete JSON message
        data = read_json_message(ser)
        if not data:
            log_message("Failed to read valid JSON message")
            return None
            
        # Extract and validate required fields
        required_fields = ['x_steps', 'y_steps', 'z_steps', 
                          'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
                          'spectrum_data']
                          
        for field in required_fields:
            if field not in data:
                log_message(f"Missing required field in JSON: {field}")
                return None
                
        # Convert spectrum data to numpy array
        try:
            spectrum = np.array(data['spectrum_data'], dtype=np.float32)
            expected_size = data['x_steps'] * data['y_steps'] * data['z_steps']
            
            if len(spectrum) != expected_size:
                log_message(f"Spectrum data size mismatch. Expected {expected_size}, got {len(spectrum)}")
                return None
                
            # Reshape the spectrum data
            spectrum = spectrum.reshape((data['x_steps'], data['y_steps'], data['z_steps']))
            
            return {
                'x': np.linspace(data['x_min'], data['x_max'], data['x_steps']),
                'y': np.linspace(data['y_min'], data['y_max'], data['y_steps']),
                'z': np.linspace(data['z_min'], data['z_max'], data['z_steps']),
                'spectrum': spectrum,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            log_message(f"Error processing spectrum data: {e}")
            return None
            
    except Exception as e:
        log_message(f"Error in read_spectrum: {e}")
        return None

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