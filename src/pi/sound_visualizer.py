import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless operation
import matplotlib.pyplot as plt
import serial
import json
import os
import time
import glob
import serial.tools.list_ports
from datetime import datetime

# Configuration
SERIAL_PORTS = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyAMA0']  # Common ports for Teensy
BAUD_RATE = 115200
OUTPUT_DIR = 'spectrum_plots'
LOG_FILE = 'spectrum_log.txt'
PLOT_INTERVAL = 1.0  # Minimum seconds between plots
MAX_PLOTS = 1000

# JSON message separator
MESSAGE_SEPARATOR = b'\n\n'  # Double newline to separate JSON messages

# Track if we've logged before in this session
_first_log = True

# Binary message header format
class MessageHeader:
    STRUCT_FORMAT = '<III'  # Little-endian, 3 unsigned ints (magic, length, checksum)
    SIZE = 12  # 3 * 4 bytes
    MAGIC = 0x4A534F4E  # 'JSON' in ASCII

def find_teensy_port():
    """Try to find the Teensy by checking common ports or listing all available"""
    # First check common ports
    for port in SERIAL_PORTS:
        if os.path.exists(port):
            log_message(f"Found potential Teensy at {port}")
            return port
    
    # If not found, list all available ports
    log_message("Teensy not found in common ports. Available ports:")
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        log_message(f"  {port.device} - {port.description}")
    
    if ports:
        return ports[0].device
    
    return None

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
    """Read a complete JSON message using binary protocol"""
    import struct
    
    start_time = time.time()
    header_data = bytearray()
    
    log_message("Waiting for message header...")
    
    # Read header
    while len(header_data) < MessageHeader.SIZE:
        if time.time() - start_time > timeout:
            log_message("Timeout waiting for message header")
            return None
            
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            header_data.extend(chunk)
            log_message(f"Read {len(chunk)} bytes of header data")
        else:
            time.sleep(0.01)
    
    # Parse header
    try:
        magic, length, checksum = struct.unpack(MessageHeader.STRUCT_FORMAT, header_data[:MessageHeader.SIZE])
    except struct.error as e:
        log_message(f"Error parsing header: {e}")
        log_message(f"Header bytes: {header_data[:MessageHeader.SIZE].hex()}")
        return None
    
    if magic != MessageHeader.MAGIC:
        log_message(f"Invalid magic number: 0x{magic:08X} (expected 0x{MessageHeader.MAGIC:08X})")
        return None
    
    log_message(f"Message header: length={length}, checksum=0x{checksum:08X}")
    
    # Read JSON data
    data = bytearray()
    bytes_read = 0
    
    log_message(f"Reading {length} bytes of JSON data...")
    
    while bytes_read < length:
        if time.time() - start_time > timeout:
            log_message(f"Timeout reading JSON data (got {bytes_read}/{length} bytes)")
            return None
            
        if ser.in_waiting > 0:
            chunk = ser.read(min(ser.in_waiting, length - bytes_read))
            data.extend(chunk)
            bytes_read += len(chunk)
            log_message(f"Read {len(chunk)} bytes of JSON data ({bytes_read}/{length} total)")
        else:
            time.sleep(0.01)
    
    # Verify checksum
    calculated_checksum = sum(data)
    if calculated_checksum != checksum:
        log_message(f"Checksum mismatch: expected 0x{checksum:08X}, got 0x{calculated_checksum:08X}")
        return None
    
    # Try to parse JSON
    try:
        json_str = data.decode('utf-8')
        log_message("Successfully decoded JSON string")
        log_message(f"JSON length: {len(json_str)} characters")
        
        # Parse the JSON
        result = json.loads(json_str)
        log_message("Successfully parsed JSON message")
        
        # Log some basic info about the data
        if isinstance(result, dict):
            log_message(f"Message keys: {list(result.keys())}")
            if 'spectrum_data' in result and isinstance(result['spectrum_data'], list):
                log_message(f"Spectrum data length: {len(result['spectrum_data'])}")
                if len(result['spectrum_data']) > 0:
                    log_message(f"First value: {result['spectrum_data'][0]}")
        
        return result
        
    except UnicodeDecodeError as e:
        log_message(f"Failed to decode JSON as UTF-8: {e}")
        log_message(f"First 100 bytes (hex): {data[:100].hex()}")
        return None
    except json.JSONDecodeError as e:
        log_message(f"Failed to parse JSON: {e}")
        log_message(f"JSON content (first 200 chars): {data[:200].decode('utf-8', errors='replace')}")
        return None
    except Exception as e:
        log_message(f"Unexpected error: {e}")
        return None

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
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find and open serial connection to Teensy
    port = find_teensy_port()
    if not port:
        log_message("Error: Could not find Teensy. Please check the connection.")
        return
        
    log_message(f"Found Teensy at {port}, connecting at {BAUD_RATE} baud...")
    
    try:
        # Open serial connection with explicit timeout and buffer settings
        ser = serial.Serial(
            port=port,
            baudrate=BAUD_RATE,
            timeout=1.0,  # Shorter timeout for faster response
            write_timeout=1.0,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        
        # Give the connection a moment to establish
        time.sleep(2)
        
        # Flush any existing data in the input buffer
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        log_message(f"Connected to {ser.name}")
        log_message("Waiting for data from Teensy...")
        
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