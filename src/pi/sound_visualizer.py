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
    
    if isinstance(message, dict):
        # Dictionary format
        x_steps = message.get('x_steps', 0) if isinstance(message.get('x_steps'), int) else 0
        y_steps = message.get('y_steps', 0) if isinstance(message.get('y_steps'), int) else 0
        checksum += x_steps ^ y_steps
        
        # Add float values
        for field in ['x_min', 'x_max', 'y_min', 'y_max']:
            if field in message and isinstance(message[field], (int, float)):
                # Convert to bytes and back to get consistent bit pattern
                float_val = float(message[field])
                checksum += struct.unpack('!I', struct.pack('!f', float_val))[0]
        
        # Add 2D spectrum data to checksum
        if 'spectrum_data' in message and isinstance(message['spectrum_data'], (list, tuple)):
            for row in message['spectrum_data']:
                if not isinstance(row, (list, tuple)):
                    continue
                for value in row:
                    if isinstance(value, (int, float)):
                        checksum += struct.unpack('!I', struct.pack('!f', float(value)))[0]
    
    return checksum & 0xFFFFFFFF  # Ensure 32-bit unsigned

def read_msgpacketizer_message(ser, timeout=5.0):
    """Read a MsgPacketizer message with start/end markers and length prefix"""
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
            elif marker != 0x1F:
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
            chunk = ser.read(ser.in_waiting)
            length_data.extend(chunk)
        else:
            time.sleep(0.01)
    
    if len(length_data) < 4:
        log_message(f"Incomplete length data: {len(length_data)} bytes")
        return None
    
    # Unpack the 4-byte big-endian length
    try:
        msg_length = struct.unpack('>I', length_data[:4])[0]
        log_message(f"Message length: {msg_length} bytes")
    except struct.error as e:
        log_message(f"Error unpacking message length: {e}")
        log_message(f"Length bytes (hex): {length_data.hex()}")
        return None
    
    # Sanity check message length
    if msg_length > MAX_MESSAGE_SIZE or msg_length == 0:
        return None
    
    # Read the message data
    data = bytearray()
    bytes_read = 0
    
    log_message(f"Reading {msg_length} bytes of MsgPack data...")
    
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
    
    # Read and verify end of message marker (0x1E)
    end_marker = ser.read(1)
    if not end_marker or end_marker[0] != 0x1E:
        log_message(f"Missing or invalid end marker. Got: {end_marker.hex() if end_marker else 'None'}")
        return None
        
    if bytes_read != msg_length:
        log_message(f"Incomplete message: expected {msg_length} bytes, got {bytes_read}")
        return None
    
    # Try to unpack the MsgPack data
    try:
        # First try to unpack as a standard message
        try:
            result = msgpack.unpackb(data, raw=False, strict_map_key=False)
            log_message("Successfully unpacked MsgPack message")
            
            # MsgPacketizer sends data as an array, convert to dict
            if isinstance(result, list):
                keys = ['x_steps', 'y_steps', 'x_min', 'x_max', 'y_min', 'y_max', 'spectrum_data', 'checksum']
                result = dict(zip(keys, result))
                log_message("Converted array to dictionary format")
            
            # Convert bytes keys to strings for easier handling if needed
            if isinstance(result, dict):
                result = {k.decode() if isinstance(k, bytes) else k: v for k, v in result.items()}
                
                # Verify checksum if present
                if 'checksum' in result:
                    # Make a copy of the result and remove checksum for calculation
                    checksum_data = result.copy()
                    received_checksum = checksum_data.pop('checksum')
                    
                    # Calculate checksum of the data
                    calculated_checksum = calculate_checksum(checksum_data)
                    
                    if received_checksum != calculated_checksum:
                        log_message(f"Checksum mismatch: received 0x{received_checksum:08X}, calculated 0x{calculated_checksum:08X}")
                        return None
                    log_message("Checksum verified successfully")
                
                return result
            
        except Exception as e:
            log_message(f"Failed to unpack MsgPack data: {e}")
            log_message(f"Raw data (hex): {data.hex()}")
            return None
    except Exception as e:
        log_message(f"Failed to process message: {e}")
        if data:
            log_message(f"First 100 bytes (hex): {data[:100].hex()}")
        import traceback
        log_message(traceback.format_exc())
        return None

def read_spectrum(ser):
    """Read a complete spectrum data message from the serial port"""
    message = read_msgpacketizer_message(ser)
    if message and 'spectrum_data' in message:
        # Add timestamp to the message
        message['timestamp'] = datetime.now()
        
        # Convert bytes keys to strings if needed
        message = {k.decode() if isinstance(k, bytes) else k: v for k, v in message.items()}
        
        # Validate spectrum data dimensions (2D array now)
        if all(key in message for key in ['x_steps', 'y_steps']):
            # Check if we have the correct number of rows (Y)
            if len(message['spectrum_data']) != message['y_steps']:
                log_message(f"Y dimension mismatch: expected {message['y_steps']} rows, got {len(message['spectrum_data'])}")
                return None
            
            # Check each row has the correct number of columns (X)
            for i, row in enumerate(message['spectrum_data']):
                if len(row) != message['x_steps']:
                    log_message(f"X dimension mismatch in row {i}: expected {message['x_steps']} columns, got {len(row)}")
                    return None
                    
        return message
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