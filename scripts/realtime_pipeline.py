import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from ayurveda.calibration import DoshaCalibrator, bandpass_filter
from pulse_core.beat_detection import WearableDetector
from pulse_core.sqi import calculate_sqi, evaluate_signal_quality
from pulse_core.feature_extraction import extract_all_features
from ayurveda.ayurveda_mapper import map_to_ayurveda
from pulse_core.sensor import load_custom_data
import warnings
from ayurveda.unified_classifier import UnifiedClassifier
import csv
from datetime import datetime

# Configuration
FS = 1000.0
CALIBRATION_DURATION_SEC = 60
INFERENCE_WINDOW_SEC = 10
SIGMA_MULTIPLIER = 1.0

def run_pipeline(data_stream, initial_fs=1000.0):
    """
    Orchestrates the calibration and continuous inference logic for tri-sensors.
    data_stream is a generator yielding triplets of (vata_value, pitta_value, kapha_value).
    """
    os.makedirs("results", exist_ok=True)
    csv_filename = f"results/nadi_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Vata_IR", "Pitta_IR", "Kapha_IR"])
    print(f"Logging raw data to {csv_filename}")
    print(f"Starting {CALIBRATION_DURATION_SEC}-second calibration phase for Vata, Pitta & Kapha sensors. Please remain still...")
    
    vata_baseline = None
    pitta_baseline = None
    kapha_baseline = None
    
    samples_collected = 0
    vata_buffer = []
    pitta_buffer = []
    kapha_buffer = []
    
    start_time = time.time()
    last_print_time = start_time
    
    # 1. Calibration Phase
    for vata_val, pitta_val, kapha_val in data_stream:
        csv_writer.writerow([time.time(), vata_val, pitta_val, kapha_val])
        vata_buffer.append(vata_val)
        pitta_buffer.append(pitta_val)
        kapha_buffer.append(kapha_val)
        samples_collected += 1
        
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Print progress every 2 seconds
        if current_time - last_print_time >= 2.0:
            print(f"Calibration progress: {min(100.0, (elapsed / CALIBRATION_DURATION_SEC) * 100):.1f}% ({samples_collected} samples at ~{samples_collected/elapsed:.0f} Hz)")
            last_print_time = current_time
            
        if elapsed >= CALIBRATION_DURATION_SEC:
            actual_fs = samples_collected / elapsed
            print(f"\nCalibration buffer full. Detected actual sampling rate: {actual_fs:.1f} Hz")
            
            # Setup calibrators
            vata_calibrator = DoshaCalibrator(fs=actual_fs, duration_sec=CALIBRATION_DURATION_SEC)
            vata_calibrator.buffer = vata_buffer
            vata_calibrator.target_samples = samples_collected
            
            pitta_calibrator = DoshaCalibrator(fs=actual_fs, duration_sec=CALIBRATION_DURATION_SEC)
            pitta_calibrator.buffer = pitta_buffer
            pitta_calibrator.target_samples = samples_collected
            
            kapha_calibrator = DoshaCalibrator(fs=actual_fs, duration_sec=CALIBRATION_DURATION_SEC)
            kapha_calibrator.buffer = kapha_buffer
            kapha_calibrator.target_samples = samples_collected
            
            print("Computing baselines...")
            vata_baseline = vata_calibrator.calibrate()
            pitta_baseline = pitta_calibrator.calibrate()
            kapha_baseline = kapha_calibrator.calibrate()
            
            if vata_baseline and pitta_baseline and kapha_baseline:
                print(f"--- VATA SENSOR (Index Finger) Baseline ---")
                print(f"  h1 Median: {vata_baseline['h1_median']:.2f} (std: {vata_baseline['h1_std']:.2f})")
                print(f"  AI Median: {vata_baseline['ai_median']:.2f} (std: {vata_baseline['ai_std']:.2f})")
                
                print(f"\n--- PITTA SENSOR (Middle Finger) Baseline ---")
                print(f"  h1 Median: {pitta_baseline['h1_median']:.2f} (std: {pitta_baseline['h1_std']:.2f})")
                print(f"  AI Median: {pitta_baseline['ai_median']:.2f} (std: {pitta_baseline['ai_std']:.2f})")
                
                print(f"\n--- KAPHA SENSOR (Ring Finger) Baseline ---")
                print(f"  h1 Median: {kapha_baseline['h1_median']:.2f} (std: {kapha_baseline['h1_std']:.2f})")
                print(f"  AI Median: {kapha_baseline['ai_median']:.2f} (std: {kapha_baseline['ai_std']:.2f})")
                
                FS = actual_fs # Update global FS for continuous phase
                break
            else:
                print("Failed to establish baseline for one or more sensors due to poor signal quality. Restarting calibration...")
                vata_buffer = []
                pitta_buffer = []
                kapha_buffer = []
                samples_collected = 0
                start_time = time.time()

    if not vata_baseline or not pitta_baseline or not kapha_baseline:
        print("Data stream ended before calibration could complete.")
        return

    print("\nStarting continuous inference mode...")
    
    # 2. Continuous Inference Phase
    vata_inference_buffer = []
    pitta_inference_buffer = []
    kapha_inference_buffer = []
    
    window_samples = int(INFERENCE_WINDOW_SEC * FS)
    detector = WearableDetector()
    
    for vata_val, pitta_val, kapha_val in data_stream:
        csv_writer.writerow([time.time(), vata_val, pitta_val, kapha_val])
        vata_inference_buffer.append(vata_val)
        pitta_inference_buffer.append(pitta_val)
        kapha_inference_buffer.append(kapha_val)
        
        if len(vata_inference_buffer) == window_samples:
            
            def process_sensor(buffer, baseline):
                raw_signal = np.array(buffer)
                try:
                    signal = bandpass_filter(raw_signal, 0.5, 10.0, FS)
                except ValueError:
                    return None, "Filter error (sampling rate too low)"
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    sqi_metrics = calculate_sqi(signal, FS)
                    sqi_class = evaluate_signal_quality(sqi_metrics['Composite_SQI'])
                    
                    if sqi_class == 'Poor':
                        return None, "Poor signal quality"
                    
                    det_result = detector.detect(signal, FS)
                    if len(det_result.systolic_peaks) >= 3:
                        det_result.confidence = 0.90 if sqi_class == 'Good' else 0.70
                        features, status = extract_all_features(signal, FS, det_result)
                        
                        if status == "SUCCESS":
                            return map_to_ayurveda(features, baseline=baseline, sigma_multiplier=SIGMA_MULTIPLIER), None
                        else:
                            return None, f"Extraction Failed: {status}"
                    else:
                        return None, "No clear beats detected"
            
            # Process all three independent sensors
            vata_report, vata_err = process_sensor(vata_inference_buffer, vata_baseline)
            pitta_report, pitta_err = process_sensor(pitta_inference_buffer, pitta_baseline)
            kapha_report, kapha_err = process_sensor(kapha_inference_buffer, kapha_baseline)
            
            # Unify with Meta-Classifier
            meta = UnifiedClassifier()
            unified_result = meta.classify(vata_report, pitta_report, kapha_report)
            
            # Terminal Output Formatting
            print("\n" + "="*55)
            print("         NADI PARIKSHA UNIFIED DIAGNOSTIC")
            print("="*55)
            print(f"Primary Dosha:    {unified_result['Primary_Dosha']}")
            print(f"Secondary Dosha:  {unified_result['Secondary_Dosha']}")
            print(f"Confidence Score: {unified_result['Confidence']}%")
            if unified_result['Confidence'] == 0:
                print(f"Reason:           {unified_result.get('Reason', 'Unknown')}")
            else:
                scores = unified_result['Scores']
                print(f"Aggregated Score: Vata: {scores['Vata']:.1f} | Pitta: {scores['Pitta']:.1f} | Kapha: {scores['Kapha']:.1f}")
            print("-" * 55)
            
            def print_individual(name, report, err):
                print(f"[{name}] ", end="")
                if report:
                    print(f"{report.get('Dominant_Dosha')} | Rate: {report.get('Vega_HeartRate_BPM', np.nan):.0f} BPM | Vol: {report.get('Akruti_Dosha')}")
                else:
                    print(f"Failed: {err}")
                    
            print_individual("Index Finger (Vata)  ", vata_report, vata_err)
            print_individual("Middle Finger (Pitta)", pitta_report, pitta_err)
            print_individual("Ring Finger (Kapha)  ", kapha_report, kapha_err)
            print("="*55 + "\n")
            
            csv_file.flush() # Ensure data is written to disk
            
            # Slide window (0 overlap for this demo)
            vata_inference_buffer = []
            pitta_inference_buffer = []
            kapha_inference_buffer = []

def simulate_stream_from_csv(filepath):
    print(f"Simulating real-time tri-stream from {filepath} (This might take a moment if the file is large)")
    sensor_data = load_custom_data(filepath)
    # Yield triplets (value, value, value) to mock three parallel sensors
    for value in sensor_data.signal:
        yield (value, value, value)

def live_stream_from_serial():
    import serial
    import serial.tools.list_ports
    BAUD_RATE = 115200
    
    print("Searching for ESP32...")
    ports = serial.tools.list_ports.comports()
    port_name = None
    for port in ports:
        if "usbserial" in port.device.lower() or "usbmodem" in port.device.lower() or "uart" in port.device.lower() or "ch340" in port.description.lower() or "cp210" in port.description.lower() or "slab" in port.description.lower():
            port_name = port.device
            break
            
    if not ports and not port_name:
        port_name = ports[0].device if ports else None
        
    if not port_name:
        print("ERROR: ESP32 not found. Check USB connection.")
        return
        
    print(f"Connecting to ESP32 on {port_name} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1.0)
    except Exception as e:
        print(f"Failed to open port {port_name}: {e}")
        return
        
    print("Connected! Reading live tri-sensor stream...")
    
    ser.reset_input_buffer()
    
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            # Expected format: "Vata:12345,Pitta:67890,Kapha:13579"
            if line.startswith("Vata:") and "Pitta:" in line and "Kapha:" in line:
                parts = line.split(",")
                vata_str = parts[0].split(":")[1]
                pitta_str = parts[1].split(":")[1]
                kapha_str = parts[2].split(":")[1]
                vata_value = float(vata_str)
                pitta_value = float(pitta_str)
                kapha_value = float(kapha_str)
                yield (vata_value, pitta_value, kapha_value)
        except Exception:
            pass # Ignore corrupt lines during startup or transmission errors

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode using CSV')
    args = parser.parse_args()
    
    if args.simulate:
        csv_file = "data/custom_radial/subject_custom_1000hz.csv"
        stream = simulate_stream_from_csv(csv_file)
    else:
        stream = live_stream_from_serial()
        
    if stream:
        run_pipeline(stream)
