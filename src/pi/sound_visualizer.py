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
    data = read_binary_message(ser)
    if not data:
        return None
    
    try:
        offset = 0
        
        # Read source count
        if len(data) < 1:
            raise ValueError("No data received")
        source_count = data[0]
        offset += 1
        
        # Calculate expected data size
        SOURCE_SIZE = 12  # 3 floats (x, y, strength) * 4 bytes each
        expected_size = 1 + (source_count * SOURCE_SIZE) + 4  # count + sources + checksum
        
        if len(data) != expected_size:
            raise ValueError(f"Invalid data length: expected {expected_size} bytes, got {len(data)}")
        
        # Read sources
        sources = []
        for _ in range(source_count):
            if offset + SOURCE_SIZE > len(data):
                raise ValueError("Unexpected end of data while reading sources")
            
            # Read x, y, strength as floats
            x = struct.unpack('>f', data[offset:offset+4])[0]; offset += 4
            y = struct.unpack('>f', data[offset:offset+4])[0]; offset += 4
            strength = struct.unpack('>f', data[offset:offset+4])[0]; offset += 4
            
            sources.append({
                'x': x,
                'y': y,
                'strength': strength
            })
        
        # Read checksum
        received_checksum = struct.unpack('>I', data[offset:offset+4])[0]
        
        # Create message dictionary
        message = {
            'sources': sources,
            'count': source_count,
            'checksum': received_checksum,
            'timestamp': datetime.now()
        }
        
        # Calculate checksum (sum of all bytes except the last 4)
        calculated_checksum = sum(data[:-4])
        
        # Verify checksum
        if received_checksum != calculated_checksum:
            log_message(f"Checksum mismatch: received 0x{received_checksum:08X}, calculated 0x{calculated_checksum:08X}")
            return None
        
        log_message(f"Successfully read {source_count} sound sources")
        return message
        
    except Exception as e:
        log_message(f"Error parsing sound sources: {e}")
        import traceback
        log_message(traceback.format_exc())
        return None

def plot_spectrum(spectrum_data, output_dir, plot_number):
    """Create and save a visualization of the spectrum"""
    try:
        # Convert the 2D array to numpy array
        spectrum_2d = np.array(spectrum_data['spectrum_data'])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot the 2D spectrum
        plt.imshow(spectrum_2d.T,
                  extent=[spectrum_data['x_min'], spectrum_data['x_max'],
                          spectrum_data['y_min'], spectrum_data['y_max']],
                  origin='lower', aspect='auto', cmap='viridis')
        
        plt.colorbar(label='Average Power (dB)')
        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.title('Sound Source Localization (Z-Averaged)')
        
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
    
    # Main loop
    plot_number = 0
    last_plot_time = time.time()
    
    try:
        while True:
            # Read sound sources
            sources_data = read_sound_sources(ser)
            
            if sources_data and sources_data['count'] > 0:
                # Filter sources by minimum strength
                filtered_sources = [s for s in sources_data['sources'] if s['strength'] >= args.min_strength]
                sources_data['sources'] = filtered_sources
                sources_data['count'] = len(filtered_sources)
                
                # Plot at specified interval if we have any sources
                current_time = time.time()
                if current_time - last_plot_time >= args.plot_interval and sources_data['count'] > 0:
                    plot_sound_sources(sources_data, args.output, plot_number)
                    plot_number += 1
                    last_plot_time = current_time
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