import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless operation
import matplotlib.pyplot as plt
import serial
import os
import time
import glob
import struct
import msgpack
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

# MessagePack configuration
MESSAGEPACK_HEADER_SIZE = 4  # 4-byte length prefix
MAX_MESSAGE_SIZE = 100000  # Sanity check for message size

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

def calculate_checksum(message):
    """Calculate checksum of the message data to verify integrity"""
    import struct
    checksum = 0
    
    # Add header values to checksum
    x_steps = message.get('x_steps', 0)
    y_steps = message.get('y_steps', 0)
    checksum += x_steps ^ y_steps
    
    # Add float values to checksum
    for field in ['x_min', 'x_max', 'y_min', 'y_max']:
        if field in message and isinstance(message[field], (int, float)):
            # Convert to bytes and back to get consistent bit pattern
            float_val = float(message[field])
            checksum += struct.unpack('>I', struct.pack('>f', float_val))[0]
    
    # Add 2D spectrum data to checksum
    if 'spectrum_data' in message and isinstance(message['spectrum_data'], (list, tuple)):
        for row in message['spectrum_data']:
            if not isinstance(row, (list, tuple)):
                continue
            for value in row:
                if isinstance(value, (int, float)):
                    # Convert to bytes and back to get consistent bit pattern
                    float_val = float(value)
                    checksum += struct.unpack('>I', struct.pack('>f', float_val))[0]
    
    return checksum & 0xFFFFFFFF  # Ensure 32-bit unsigned

def read_binary_message(ser, timeout=5.0):
    """Read a binary message with start/end markers and length prefix"""
    start_time = time.time()
    
    # Wait for start of message marker (0x1F)
    while True:
        if time.time() - start_time > timeout:
            log_message("Timeout waiting for start of message")
            return None
            
        if ser.in_waiting > 0:
            marker = ser.read(1)[0]
            if marker == 0x1F:  # Start of message marker
                break
            else:
                log_message(f"Unexpected byte 0x{marker:02x}, expected start marker 0x1F")
                # Skip this byte and continue looking for start marker
                continue
    
    # Read the 4-byte length prefix (big-endian)
    length_data = bytearray()
    while len(length_data) < 4:  # 4-byte length prefix
        if time.time() - start_time > timeout:
            log_message("Timeout waiting for message length")
            return None
            
        if ser.in_waiting > 0:
            chunk = ser.read(1)  # Read one byte at a time for length
            length_data.extend(chunk)
        else:
            time.sleep(0.001)
    
    # Unpack the 4-byte big-endian length
    try:
        msg_length = struct.unpack('>I', length_data)[0]
        log_message(f"Message length: {msg_length} bytes")
    except struct.error as e:
        log_message(f"Error unpacking message length: {e}")
        log_message(f"Length bytes (hex): {length_data.hex()}")
        return None
    
    # Sanity check message length
    if msg_length > MAX_MESSAGE_SIZE or msg_length == 0:
        log_message(f"Invalid message length: {msg_length}")
        return None
    
    # Read the message data
    data = bytearray()
    bytes_read = 0
    
    log_message(f"Reading {msg_length} bytes of binary data...")
    
    while bytes_read < msg_length:
        if time.time() - start_time > timeout:
            log_message(f"Timeout reading message data. Got {bytes_read} of {msg_length} bytes")
            return None
            
        chunk = ser.read(min(ser.in_waiting, msg_length - bytes_read))
        if not chunk:
            time.sleep(0.001)
            continue
            
        data.extend(chunk)
        bytes_read += len(chunk)
        
        if msg_length > 1000 and bytes_read % 1000 == 0:  # Log progress for large messages
            log_message(f"Read {bytes_read}/{msg_length} bytes ({bytes_read/msg_length*100:.1f}%)")
    
    if bytes_read != msg_length:
        log_message(f"Incomplete message: expected {msg_length} bytes, got {bytes_read}")
        return None
    
    try:
        # Read and verify end of message marker (0x1E)
        end_marker = ser.read(1)
        if not end_marker or end_marker[0] != 0x1E:
            log_message(f"Missing or invalid end marker. Got: {end_marker.hex() if end_marker else 'None'}")
            return None
            
        return data
    except Exception as e:
        log_message(f"Failed to process message: {e}")
        if data:
            log_message(f"First 100 bytes (hex): {data[:100].hex()}")
        import traceback
        log_message(traceback.format_exc())
        return None

def read_sound_sources(ser):
    """Read and parse sound sources data from the serial port"""
    try:
        # Read a line from serial
        line = ser.readline().decode('ascii', errors='ignore').strip()
        if not line:
            return None
            
        # Skip non-source lines
        if not line.startswith('Sources['):
            log_message(f"Skipping line: {line}")
            return None
            
        # Parse the source count
        try:
            # Extract the count from 'Sources[2]: 0:(...' -> '2'
            count_str = line.split('[')[1].split(']')[0]
            source_count = int(count_str)
            
            # Extract the sources part after ']: '
            sources_part = line.split(']: ', 1)[1]
            
            # Parse each source
            sources = []
            for source_str in sources_part.split(' | '):
                # Format: '0:(x,y,str)'
                if not source_str or ':' not in source_str or '(' not in source_str:
                    continue
                    
                # Extract coordinates and strength
                coords_str = source_str.split('(', 1)[1].rstrip(')')
                x_str, y_str, strength_str = coords_str.split(',')
                
                sources.append({
                    'x': float(x_str),
                    'y': float(y_str),
                    'strength': float(strength_str)
                })
            
            # Create message dictionary
            message = {
                'sources': sources,
                'count': source_count,
                'timestamp': datetime.now()
            }
            
            # Verify we got the expected number of sources
            if len(sources) != source_count:
                log_message(f"Warning: Expected {source_count} sources, got {len(sources)}")
            
            log_message(f"Parsed {len(sources)} sound sources")
            return message
            
        except (ValueError, IndexError) as e:
            log_message(f"Error parsing line '{line}': {e}")
            return None
            
    except Exception as e:
        log_message(f"Error reading from serial: {e}")
        import traceback
        log_message(traceback.format_exc())
        return None

def plot_spectrum(sources_data, output_dir, plot_number):
    """Create and save a visualization of the sound sources"""
    try:
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create a scatter plot of the sources
        if sources_data['count'] > 0:
            x = [s['x'] for s in sources_data['sources']]
            y = [s['y'] for s in sources_data['sources']]
            strengths = [s['strength'] for s in sources_data['sources']]
            
            # Scale marker size by strength (with some reasonable limits)
            sizes = [100 + s * 500 for s in strengths]  # Scale factor can be adjusted
            
            # Create scatter plot
            scatter = plt.scatter(x, y, s=sizes, c=strengths, 
                                cmap='viridis', alpha=0.7,
                                vmin=0, vmax=1)  # Normalize strength to 0-1
            
            # Add colorbar
            plt.colorbar(scatter, label='Source Strength')
            
            # Add source numbers
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.text(xi, yi, str(i), ha='center', va='center', 
                        color='white' if strengths[i] > 0.5 else 'black')
        
        # Set plot limits (adjust these based on your setup)
        plt.xlim(-1000, 1000)  # mm
        plt.ylim(-1000, 1000)  # mm
        
        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.title('Sound Source Localization')
        plt.grid(True, alpha=0.3)
        
        # Add timestamp
        timestamp = sources_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        plt.figtext(0.5, 0.01, f'Last Update: {timestamp} | {sources_data["count"]} sources', 
                   ha='center', fontsize=8, style='italic')
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sources_{plot_number:04d}.png')
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
        import traceback
        log_message(traceback.format_exc())
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
    
    try:
        # Open serial connection
        ser = serial.Serial(port, baudrate=115200, timeout=1.0)
        log_message(f"Connected to {port} at 115200 baud")
    except serial.SerialException as e:
        log_message(f"Error opening serial port {port}: {e}")
        return

    # Main loop
    plot_number = 0
    last_plot_time = time.time()
    
    try:
        while True:
            # Read sound sources
            sources_data = read_sound_sources(ser)
            
            if sources_data and sources_data['count'] > 0:
                # Filter sources by minimum strength
                filtered_sources = [s for s in sources_data['sources'] if s['strength'] >= 0.0]
                sources_data['sources'] = filtered_sources
                sources_data['count'] = len(filtered_sources)
                
                # Plot at specified interval if we have any sources
                current_time = time.time()
                if current_time - last_plot_time >= 1.0 and sources_data['count'] > 0:
                    plot_spectrum(sources_data, OUTPUT_DIR, plot_number)
                    plot_number += 1
                    last_plot_time = current_time
                    
                    # Clean up old plots
                    cleanup_old_plots(OUTPUT_DIR, MAX_PLOTS)
    
    except KeyboardInterrupt:
        log_message("Spectrum visualizer stopped by user")
    except Exception as e:
        log_message(f"Unexpected error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            log_message("Closed serial connection")
        log_message("Spectrum visualizer stopped")

if __name__ == "__main__":
    main()