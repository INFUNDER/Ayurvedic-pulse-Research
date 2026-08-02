import serial
import serial.tools.list_ports
import csv
import time
import os
import signal
import sys

# --- Configuration ---
BAUD_RATE = 921600
CSV_FILENAME = "subject_custom_1000hz.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'custom_radial')
os.makedirs(DATA_DIR, exist_ok=True)

recording = True

def signal_handler(sig, frame):
    global recording
    print("\nStopping recording gracefully...")
    recording = False

signal.signal(signal.SIGINT, signal_handler)

def find_esp32_port():
    """Auto-detect the ESP32 serial port (macOS typical paths)."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "usbserial" in port.device.lower() or "usbmodem" in port.device.lower() or "uart" in port.device.lower() or "ch340" in port.description.lower() or "cp210" in port.description.lower():
            return port.device
    if ports:
        return ports[0].device
    return None

def main():
    global recording
    port = find_esp32_port()
    if not port:
        print("ERROR: ESP32 not found. Check USB connection.")
        return
        
    print(f"Connecting to ESP32 on {port} at {BAUD_RATE} baud...")
    
    try:
        # Open port for high-speed logging
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"Failed to open port {port}: {e}")
        return
        
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)
    
    # We remove matplotlib because at 1000Hz (921600 baud), python's plotting overhead 
    # will cause serial buffer overflows and drop samples. We strictly log to disk.
    
    print(f"\nRecording started. Saving to: {csv_path}")
    print("Press Ctrl+C to stop recording.")
    
    start_time = time.time()
    samples_collected = 0
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp_ms", "Red", "IR"])
        
        while recording:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line and not line.startswith("System Ready"):
                    parts = line.split(',')
                    if len(parts) == 3:
                        writer.writerow([parts[0], parts[1], parts[2]])
                        samples_collected += 1
                        
                        # Print progress every 1000 samples (~1 second)
                        if samples_collected % 1000 == 0:
                            elapsed = time.time() - start_time
                            print(f"Collected {samples_collected} samples... ({samples_collected/elapsed:.1f} Hz)")
                            
            except Exception:
                pass # Ignore corrupt lines due to baud mismatch during startup
                
    ser.close()
    print(f"Recording stopped. Saved {samples_collected} samples.")

if __name__ == "__main__":
    main()
