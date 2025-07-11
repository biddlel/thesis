#!/usr/bin/env python3
import serial
import serial.tools.list_ports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from flask import Flask, send_file, Response
import io
import threading

# Configuration
SERIAL_BAUDRATE = 115200
PLOT_UPDATE_INTERVAL = 0.1  # seconds
PLOT_SIZE = (10, 8)  # inches
PLOT_DPI = 100
SERIAL_TIMEOUT = 1

# Initialize Flask app
app = Flask(__name__)

# Global variables for data sharing between threads
latest_position = (0, 0)
plot_lock = threading.Lock()
plot_img = None

# Find and initialize serial port
def init_serial():
    # List all available ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(f"Found port: {p.device} - {p.description}")
        if 'ttyACM' in p.device or 'ttyUSB' in p.device:
            try:
                ser = serial.Serial(p.device, baudrate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
                print(f"Connected to {p.device}")
                return ser
            except Exception as e:
                print(f"Failed to open {p.device}: {e}")
    raise Exception("No suitable serial port found")

def parse_serial_data(line):
    try:
        # Expected format: "Estimated Coords (mm): X=123.4, Y=567.8"
        if line.startswith('Estimated Coords'):
            parts = line.split()
            x = float(parts[3].split('=')[1][:-1])  # Remove trailing comma
            y = float(parts[4].split('=')[1])
            return x, y
    except Exception as e:
        print(f"Error parsing line: {e}")
    return None

def update_plot(x, y):
    global plot_img
    
    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    
    # Create a grid
    x_range = np.linspace(-500, 500, 100)
    y_range = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Create a simple plot with the point
    plt.scatter([x], [y], c='red', s=200, zorder=5, label='Sound Source')
    plt.scatter([0], [0], c='blue', s=100, marker='s', label='Microphone Array')
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Sound Source Localization')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.xlim(-600, 600)
    plt.ylim(-600, 600)
    plt.legend()
    
    # Add crosshairs at the current position
    plt.axvline(x, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y, color='red', linestyle='--', alpha=0.3)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    with plot_lock:
        plot_img = buf.read()
    
    plt.close()

# Background thread to read from serial and update plot
def serial_reader():
    global latest_position
    
    ser = None
    while True:
        try:
            if ser is None:
                ser = init_serial()
            
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    result = parse_serial_data(line)
                    if result:
                        x, y = result
                        latest_position = (x, y)
                        print(f"Updated position: X={x:.1f}, Y={y:.1f}")
                        update_plot(x, y)
        except Exception as e:
            print(f"Serial error: {e}")
            if ser:
                ser.close()
                ser = None
            time.sleep(2)  # Wait before retrying
        time.sleep(0.01)  # Small delay to prevent busy waiting

# Flask routes
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sound Source Localization</title>
        <meta http-equiv="refresh" content="0.5">
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            img { max-width: 90%; height: auto; border: 1px solid #ddd; }
            .container { margin: 20px; }
            .coords { 
                font-size: 1.5em; 
                margin: 20px 0; 
                padding: 10px; 
                background-color: #f0f0f0; 
                display: inline-block;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sound Source Localization</h1>
            <div class="coords">
                Latest position: X={x:.1f}mm, Y={y:.1f}mm
            </div>
            <div>
                <img src="/plot" alt="Sound source position">
            </div>
        </div>
    </body>
    </html>
    """.format(x=latest_position[0], y=latest_position[1])

@app.route('/plot')
def plot():
    with plot_lock:
        if plot_img is None:
            return "No plot available yet", 404
        return Response(plot_img, mimetype='image/png')

def main():
    # Start serial reader thread
    serial_thread = threading.Thread(target=serial_reader, daemon=True)
    serial_thread.start()
    
    # Start Flask web server
    print("Starting web server at http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
