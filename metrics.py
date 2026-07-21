import numpy as np
try:
    import neurokit2 as nk
except ImportError:
    pass

def extract_ecg_r_peaks(ecg_signal, fs):
    """
    Extract R-peaks from an ECG signal using NeuroKit2.
    Returns array of peak indices.
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        return np.array([])
        
    try:
        # Clean the ECG signal
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
        
        # Find R-peaks
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        r_peaks = info['ECG_R_Peaks']
        return r_peaks
    except Exception as e:
        print(f"Error extracting ECG peaks: {e}")
        return np.array([])


def evaluate_beat_detection(detected_ppg_peaks, reference_ecg_peaks, ppg_fs, ecg_fs, ptt_min=0.0, ptt_max=0.4):
    """
    Evaluates PPG beat detection against ground truth ECG R-peaks.
    Allows a physiological Pulse Transit Time (PTT) delay window (e.g. 0 to 400ms).
    
    Args:
        detected_ppg_peaks: list of detected systolic peaks in PPG (indices relative to ppg_fs)
        reference_ecg_peaks: list of ground truth R-peaks in ECG (indices relative to ecg_fs)
        ppg_fs: PPG sampling rate
        ecg_fs: ECG sampling rate
        ptt_min: Minimum valid PTT (seconds). Distance from ECG R-peak to PPG Systolic peak.
        ptt_max: Maximum valid PTT (seconds).
        
    Returns:
        metrics: dict of Sensitivity, Precision, F1, MAE, Std Dev
    """
    if len(reference_ecg_peaks) == 0:
        return {'Sensitivity': 0, 'Precision': 0, 'F1': 0, 'MAE_ms': 0, 'Std_Dev_ms': 0, 'TP': 0, 'FP': len(detected_ppg_peaks), 'FN': 0}
        
    # Convert peaks to seconds for unified comparison
    ppg_times = np.array(detected_ppg_peaks) / ppg_fs
    ecg_times = np.array(reference_ecg_peaks) / ecg_fs
    
    tp = 0
    fn = 0
    fp = 0
    errors = []
    
    # Greedy matching logic
    matched_ppg = set()
    
    for e_time in ecg_times:
        valid_ppg = [p for p in ppg_times if e_time + ptt_min <= p <= e_time + ptt_max and p not in matched_ppg]
        
        if len(valid_ppg) > 0:
            p_time = valid_ppg[0] # Take the earliest valid peak
            matched_ppg.add(p_time)
            tp += 1
            errors.append(p_time - e_time)
        else:
            fn += 1
            
    fp = len(ppg_times) - len(matched_ppg)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Error in ms
    errors_ms = np.array(errors) * 1000
    mae = np.mean(errors_ms) if len(errors_ms) > 0 else 0
    std_dev = np.std(errors_ms) if len(errors_ms) > 0 else 0
    
    # Calculate ground truth HR
    true_hr = np.nan
    if len(ecg_times) > 1:
        true_hr = 60.0 / np.mean(np.diff(ecg_times))
        
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'F1': f1,
        'MAE_ms': mae,
        'Std_Dev_ms': std_dev,
        'Errors_ms': errors_ms.tolist(),
        'True_HR': true_hr
    }
