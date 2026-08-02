import numpy as np
import time
from calibration import DoshaCalibrator, bandpass_filter
from beat_detection import WearableDetector
from sqi import calculate_sqi, evaluate_signal_quality
from feature_extraction import extract_all_features
from ayurveda_mapper import map_to_ayurveda
from sensor import load_custom_data
import warnings

# Configuration
FS = 1000.0
CALIBRATION_DURATION_SEC = 60
INFERENCE_WINDOW_SEC = 10
SIGMA_MULTIPLIER = 1.0

def run_pipeline(data_stream, initial_fs=1000.0):
    """
    Orchestrates the calibration and continuous inference logic for dual sensors.
    data_stream is a generator yielding tuples of (vata_value, pitta_value).
    """
    print(f"Starting {CALIBRATION_DURATION_SEC}-second calibration phase for both Vata & Pitta sensors. Please remain still...")
    
    vata_baseline = None
    pitta_baseline = None
    
    samples_collected = 0
    vata_buffer = []
    pitta_buffer = []
    
    start_time = time.time()
    last_print_time = start_time
    
    # 1. Calibration Phase
    for vata_val, pitta_val in data_stream:
        vata_buffer.append(vata_val)
        pitta_buffer.append(pitta_val)
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
            
            print("Computing baselines...")
            vata_baseline = vata_calibrator.calibrate()
            pitta_baseline = pitta_calibrator.calibrate()
            
            if vata_baseline and pitta_baseline:
                print(f"--- VATA SENSOR (Index Finger) Baseline ---")
                print(f"  h1 Median: {vata_baseline['h1_median']:.2f} (std: {vata_baseline['h1_std']:.2f})")
                print(f"  AI Median: {vata_baseline['ai_median']:.2f} (std: {vata_baseline['ai_std']:.2f})")
                
                print(f"\n--- PITTA SENSOR (Middle Finger) Baseline ---")
                print(f"  h1 Median: {pitta_baseline['h1_median']:.2f} (std: {pitta_baseline['h1_std']:.2f})")
                print(f"  AI Median: {pitta_baseline['ai_median']:.2f} (std: {pitta_baseline['ai_std']:.2f})")
                
                FS = actual_fs # Update global FS for continuous phase
                break
            else:
                print("Failed to establish baseline for one or both sensors due to poor signal quality. Restarting calibration...")
                vata_buffer = []
                pitta_buffer = []
                samples_collected = 0
                start_time = time.time()

    if not vata_baseline or not pitta_baseline:
        print("Data stream ended before calibration could complete.")
        return

    print("\nStarting continuous inference mode...")
    
    # 2. Continuous Inference Phase
    vata_inference_buffer = []
    pitta_inference_buffer = []
    
    window_samples = int(INFERENCE_WINDOW_SEC * FS)
    detector = WearableDetector()
    
    for vata_val, pitta_val in data_stream:
        vata_inference_buffer.append(vata_val)
        pitta_inference_buffer.append(pitta_val)
        
        if len(vata_inference_buffer) == window_samples:
            
            def process_sensor(sensor_name, buffer, baseline):
                raw_signal = np.array(buffer)
                signal = bandpass_filter(raw_signal, 0.5, 10.0, FS)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    sqi_metrics = calculate_sqi(signal, FS)
                    sqi_class = evaluate_signal_quality(sqi_metrics['Composite_SQI'])
                    
                    print(f"\n========== {sensor_name} 10s Window ==========")
                    if sqi_class == 'Poor':
                        print("[Warning] Poor signal quality detected. Artifact rejected. Please stay still.")
                    else:
                        det_result = detector.detect(signal, FS)
                        if len(det_result.systolic_peaks) >= 3:
                            det_result.confidence = 0.90 if sqi_class == 'Good' else 0.70
                            features, status = extract_all_features(signal, FS, det_result)
                            
                            if status == "SUCCESS":
                                ayur = map_to_ayurveda(features, baseline=baseline, sigma_multiplier=SIGMA_MULTIPLIER)
                                print(f"Dominant Dosha: {ayur.get('Dominant_Dosha', 'N/A')}")
                                scores = ayur.get('Scores', {})
                                print(f"Scores: Vata: {scores.get('Vata', 0)}, Pitta: {scores.get('Pitta', 0)}, Kapha: {scores.get('Kapha', 0)}")
                                print(f"Vega (Rate): {ayur.get('Vega_HeartRate_BPM', np.nan):.2f} BPM -> {ayur.get('Vega_Dosha', 'N/A')}")
                                print(f"Gati (Movement): -> {ayur.get('Gati_Dosha', 'N/A')}")
                                print(f"Tala (Rhythm): CVNN={ayur.get('Tala_CVNN', np.nan):.2f}% -> {ayur.get('Tala_Dosha', 'N/A')}")
                                print(f"Bala (Force): AI={ayur.get('Bala_Augmentation_Index', np.nan):.2f} -> {ayur.get('Bala_Dosha', 'N/A')}")
                                print(f"Kathinya (Stiffness): SI={ayur.get('Kathinya_Stiffness_Index', np.nan):.2f} -> {ayur.get('Kathinya_Dosha', 'N/A')}")
                                print(f"Akruti (Volume): h1={ayur.get('Akruti_h1', np.nan):.2f} -> {ayur.get('Akruti_Dosha', 'N/A')}")
                            else:
                                print(f"Extraction Failed: {status}")
                        else:
                            print("No clear beats detected in this window.")
            
            # Process both independent sensors
            process_sensor("VATA SENSOR (Index Finger)", vata_inference_buffer, vata_baseline)
            process_sensor("PITTA SENSOR (Middle Finger)", pitta_inference_buffer, pitta_baseline)
            
            # Slide window (0 overlap for this demo)
            vata_inference_buffer = []
            pitta_inference_buffer = []

def simulate_stream_from_csv(filepath):
    print(f"Simulating real-time dual stream from {filepath} (This might take a moment if the file is large)")
    sensor_data = load_custom_data(filepath)
    # Yield tuples (value, value) to mock two parallel sensors
    for value in sensor_data.signal:
        yield (value, value)

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
        
    print("Connected! Reading live dual stream...")
    
    ser.reset_input_buffer()
    
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            # Expected format: "Vata:12345,Pitta:67890"
            if line.startswith("Vata:") and "Pitta:" in line:
                parts = line.split(",")
                vata_str = parts[0].split(":")[1]
                pitta_str = parts[1].split(":")[1]
                vata_value = float(vata_str)
                pitta_value = float(pitta_str)
                yield (vata_value, pitta_value)
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
