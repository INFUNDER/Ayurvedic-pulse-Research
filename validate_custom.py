import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sensor import load_custom_data
from noise_removal import remove_noise
from sqi import calculate_sqi, evaluate_signal_quality
from beat_detection import detect_beats
from feature_extraction import extract_all_features

def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'custom_radial', 'subject_custom_001.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    print(f"Loading custom data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Jitter Analysis
    ts = df['Timestamp_ms'].values
    dt = np.diff(ts)
    median_dt = np.median(dt)
    fs = 1000.0 / median_dt if median_dt > 0 else 100.0
    
    print(f"\n--- Acquisition Diagnostics ---")
    print(f"Calculated Median dt: {median_dt:.1f} ms")
    print(f"Empirical Sampling Rate: {fs:.1f} Hz")
    print(f"Max Jitter (dt > 1.5*median): {np.sum(dt > 1.5*median_dt)} occurrences")
    
    # Raw IR for amplitude/saturation checks (invert as we do for PPG processing)
    raw_ir = -df['IR'].values
    
    # 2. Transient Rejection (Discard first 5 seconds)
    print("\nDiscarding first 5 seconds (sensor settling/placement transient)...")
    ignore_samples = int(5.0 * fs)
    if len(raw_ir) > ignore_samples + int(10 * fs): # ensure we have enough data left
        raw_ir = raw_ir[ignore_samples:]
        ts_valid = ts[ignore_samples:]
    else:
        print("Warning: Recording too short to discard 5 seconds. Using entire signal.")
        ts_valid = ts
        
    # Re-wrap as SensorData for downstream pipeline
    from sensor import SensorData
    sensor_data = SensorData(
        signal=raw_ir,
        fs=fs,
        reference_ecg=None,
        reference_ecg_fs=None,
        label="Custom Hardware (Truncated)"
    )
    
    print(f"Remaining signal length: {len(raw_ir)} samples ({len(raw_ir)/fs:.1f} seconds)")
    
    # AC/DC Ratio calculation
    dc_level = np.abs(np.mean(raw_ir))
    ac_level = np.std(raw_ir) # Rough AC approximation
    print(f"AC Amplitude / DC Baseline: {(ac_level/dc_level)*100:.3f}%" if dc_level > 0 else "N/A")
    
    # 3. Process Data
    print("\nApplying filters (Bandpass 0.5 - 8.0 Hz)...")
    clean_sig, trim = remove_noise(sensor_data.signal, fs, use_lp=True, use_hp=True)
    clean_sig = clean_sig[trim:-trim]
    
    sqi_metrics = calculate_sqi(clean_sig, fs)
    sqi_score = sqi_metrics['Composite_SQI']
    sqi_class = evaluate_signal_quality(sqi_score)
    
    print(f"\nComposite SQI: {sqi_score:.3f} -> {sqi_class}")
    
    print("Routing signal through Research Platform v1.0...")
    det_result = detect_beats(clean_sig, fs, mode="auto", sqi_class=sqi_class, sqi_score=sqi_score)
    peaks = det_result.systolic_peaks
    
    print(f"Detector Used: {det_result.detector_used} (Confidence: {det_result.confidence})")
    
    feats, _ = extract_all_features(clean_sig, fs, det_result)
    if 'Heart_Rate_BPM' in feats:
        print(f"Heart Rate: {feats['Heart_Rate_BPM']:.1f} BPM")
        
    # 4. Generate Diagnostic Plot
    print("\nGenerating Diagnostic Plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot A: Jitter (Delta T)
    axes[0].plot(ts[1:] / 1000.0, dt, color='purple', alpha=0.7)
    axes[0].axhline(median_dt, color='red', linestyle='--', label=f'Median dt = {median_dt:.1f}ms')
    axes[0].set_title('Inter-sample Timing (Jitter Analysis)')
    axes[0].set_ylabel('Delta T (ms)')
    axes[0].legend()
    
    # Plot B: Raw Unfiltered Signal (after truncation)
    t_valid = (ts_valid - ts_valid[0]) / 1000.0
    axes[1].plot(t_valid, raw_ir, color='blue')
    axes[1].set_title('Raw IR Signal (Post-Transient)')
    axes[1].set_ylabel('Raw ADC Value (Inverted)')
    
    # Plot C: Processed Signal and Peaks
    t_clean = np.arange(len(clean_sig)) / fs
    axes[2].plot(t_clean, clean_sig, color='black')
    if len(peaks) > 0:
        axes[2].scatter(peaks / fs, clean_sig[peaks], color='red', marker='x', s=100, label='Detected Beats')
    axes[2].set_title(f'Filtered Signal & Detector Output ({det_result.detector_used})')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_custom_subject.png')
    plt.savefig(plot_path, dpi=200)
    print(f"Saved highly detailed diagnostic plot to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
