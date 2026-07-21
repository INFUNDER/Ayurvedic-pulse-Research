import os
import time
import tracemalloc
import pandas as pd
import numpy as np
import scipy.signal
from glob import glob
import matplotlib.pyplot as plt
import sys
import platform
import subprocess

import heartpy as hp
import neurokit2 as nk

from sensor import load_bidmc_data, load_mimic_data, load_wesad_data, load_dalia_data
from noise_removal import remove_noise
from beat_detection import detect_beats
from feature_extraction import extract_all_features
from metrics import extract_ecg_r_peaks, evaluate_beat_detection
from noise_robustness import (add_gaussian_noise, add_baseline_wander, 
                              add_powerline_interference, add_motion_artifact)
from sqi import calculate_sqi, evaluate_signal_quality

WINDOW_SEC = 300 # 5 minutes
FAILURE_DIR = "failure_analysis"
os.makedirs(FAILURE_DIR, exist_ok=True)

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown/Not a Git repository"

def save_reproducibility_log():
    with open('reproducibility_log.txt', 'w') as f:
        f.write("=== Reproducibility Log ===\n")
        f.write(f"OS: {platform.system()} {platform.release()} ({platform.version()})\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Pandas Version: {pd.__version__}\n")
        f.write(f"NumPy Version: {np.__version__}\n")
        f.write(f"SciPy Version: {scipy.__version__}\n")
        f.write(f"HeartPy Version: {hp.__version__ if hasattr(hp, '__version__') else 'Unknown'}\n")
        f.write(f"NeuroKit2 Version: {nk.__version__}\n")
        f.write(f"Git Commit: {get_git_commit()}\n")
        f.write("\n=== Random Seeds ===\n")
        f.write("Noise generation uses static seeds within `noise_robustness.py` (np.random.seed).\n")
    print("Saved reproducibility_log.txt")

def plot_failure(signal, fs, detected_peaks, true_peaks, title, filename, sqi_class, metrics, rejection_reason="None"):
    n_samples = min(int(15 * fs), len(signal))
    t = np.arange(n_samples) / fs
    sig_snippet = signal[:n_samples]
    
    det_snippet = [p for p in detected_peaks if p < n_samples]
    true_snippet = [p for p in true_peaks if p < n_samples]
    
    fn, fp = metrics['FN'], metrics['FP']
    
    # Advanced taxonomy classification
    if rejection_reason != "None":
        fail_type = rejection_reason
    elif fn > fp and np.var(signal) > 1.5:
        fail_type = "Missed low-amplitude beats"
    elif fp > fn and sqi_class in ['Moderate', 'Poor']:
        fail_type = "Motion false positives"
    elif fp > fn and len(detected_peaks) > 1 and np.mean(np.diff(detected_peaks))/fs < 0.3:
        fail_type = "Double detections"
    elif fn > fp and len(detected_peaks) == 0:
        fail_type = "Signal Rejected"
    else:
        fail_type = "Baseline drift"
        
    full_title = f"{title}\n{fail_type}"
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, sig_snippet, label='PPG Signal', color='black', alpha=0.7)
    plt.scatter(np.array(true_snippet)/fs, sig_snippet[true_snippet], color='green', marker='o', s=100, label='True ECG R-Peak')
    plt.scatter(np.array(det_snippet)/fs, sig_snippet[det_snippet], color='red', marker='x', s=100, label='Detected Peak')
    
    plt.title(full_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FAILURE_DIR, filename), dpi=150)
    plt.close()
    
    return fail_type

def get_5_min_window(sensor_data):
    if sensor_data.signal is None or len(sensor_data.signal) == 0:
        return sensor_data
    fs = sensor_data.fs
    n_samples = int(WINDOW_SEC * fs)
    if len(sensor_data.signal) <= n_samples:
        return sensor_data
    start_idx = len(sensor_data.signal) // 2 - n_samples // 2
    end_idx = start_idx + n_samples
    sensor_data.signal = sensor_data.signal[start_idx:end_idx]
    if sensor_data.reference_ecg is not None and sensor_data.reference_ecg_fs is not None:
        ecg_n = int(WINDOW_SEC * sensor_data.reference_ecg_fs)
        ecg_start = len(sensor_data.reference_ecg) // 2 - ecg_n // 2
        sensor_data.reference_ecg = sensor_data.reference_ecg[ecg_start:ecg_start + ecg_n]
    return sensor_data

def benchmark_dataset(data_generator, dataset_name):
    print(f"\n--- Benchmarking {dataset_name} ---")
    results = []
    error_results = []
    
    detectors = ['ours_auto', 'heartpy', 'elgendi']
    sampling_freqs = [25, 50, 125]
    
    for sensor_data in data_generator:
        try:
            print(f"Processing {sensor_data.label}...")
            sensor_data = get_5_min_window(sensor_data)
            ecg_peaks = extract_ecg_r_peaks(sensor_data.reference_ecg, sensor_data.reference_ecg_fs)
            
            def evaluate_condition(sig, fs, cond_name, is_clean=False):
                cond_results = []
                clean_sig, trim_samples = remove_noise(sig, fs, use_lp=True, use_hp=True)
                clean_sig = clean_sig[trim_samples:-trim_samples]
                
                sqi_metrics = calculate_sqi(clean_sig, fs)
                sqi_score = sqi_metrics['Composite_SQI']
                sqi_class = evaluate_signal_quality(sqi_score)
                
                for det in detectors:
                    start_time = time.time()
                    if det == 'ours_auto' and is_clean: tracemalloc.start()
                    
                    peaks = np.array([])
                    det_result = None
                    try:
                        if det == 'ours_auto':
                            det_result = detect_beats(clean_sig, fs, mode="auto", sqi_class=sqi_class, sqi_score=sqi_score)
                            peaks = det_result.systolic_peaks
                        elif det == 'heartpy':
                            if not is_clean and not cond_name.startswith('Downsampled'):
                                raise Exception("Skipping HeartPy on noise (infinite loop risk)")
                            wd, _ = hp.process(clean_sig, sample_rate=fs)
                            peaks = np.array([p for p in wd['peaklist'] if p > 0])
                        elif det == 'elgendi':
                            info = nk.ppg_findpeaks(clean_sig, sampling_rate=fs, method='elgendi')
                            peaks = info['PPG_Peaks']
                    except Exception as e:
                        pass
                    
                    metrics = evaluate_beat_detection(peaks, ecg_peaks, fs, sensor_data.reference_ecg_fs)
                    fail_type = "None"
                    
                    if det == 'ours_auto':
                        for err in metrics['Errors_ms']:
                            error_results.append({
                                'Dataset': dataset_name,
                                'Subject': sensor_data.label,
                                'Condition': cond_name,
                                'Error_ms': err
                            })
                        if metrics['F1'] < 0.50 or det_result.confidence == 0.0:
                            fail_title = f"{dataset_name} {sensor_data.label} - {cond_name} (F1: {metrics['F1']:.2f})"
                            fail_filename = f"failure_{dataset_name}_{sensor_data.label.replace(' ', '_')}_{cond_name}.png"
                            fail_type = plot_failure(clean_sig, fs, peaks, (np.array(ecg_peaks) / sensor_data.reference_ecg_fs * fs).astype(int), fail_title, fail_filename, sqi_class, metrics, det_result.rejection_reason)
                    
                    feats = {}
                    feat_status = {}
                    runtime = time.time() - start_time
                    mem = np.nan
                    
                    if det == 'ours_auto':
                        try:
                            if det_result is not None:
                                feats, feat_status = extract_all_features(clean_sig, fs, det_result)
                        except Exception as e:
                            pass
                            
                        if is_clean:
                            _, mem_peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            mem = mem_peak / 1024 / 1024
                            
                    res_dict = {
                        'Dataset': dataset_name,
                        'Subject': sensor_data.label,
                        'Condition': cond_name,
                        'Detector': det,
                        'F1': metrics['F1'],
                        'MAE_ms': metrics['MAE_ms'],
                        'Runtime_s': runtime if (det == 'ours_auto' and is_clean) else np.nan,
                        'Memory_MB': mem,
                        'True_HR': metrics.get('True_HR', np.nan),
                        'Composite_SQI': sqi_metrics['Composite_SQI'],
                        'SPR': sqi_metrics['SPR'],
                        'Kurtosis': sqi_metrics['Kurtosis'],
                        'ACF_Peak': sqi_metrics['ACF_Peak'],
                        'SQI_Class': sqi_class,
                        'Confidence': getattr(det_result, 'confidence', 0.0),
                        'Detector_Used': getattr(det_result, 'detector_used', 'none'),
                        'Failure_Type': fail_type
                    }
                    
                    if det == 'ours_auto':
                        # Track feature success status strings
                        for f_key in ['Heart_Rate_BPM', 'HRV_RMSSD', 'HRV_SDNN', 'Sample_Entropy', 'Mean_Width_at_50', 'APG_Aging_Index']:
                            res_dict[f'Status_{f_key}'] = feat_status.get(f_key, "Unknown")
                            
                    cond_results.append(res_dict)
                return cond_results
            
            clean_res = evaluate_condition(sensor_data.signal, sensor_data.fs, 'Clean', is_clean=True)
            results.extend(clean_res)
            
            noise_conditions = {
                'Gaussian_20dB': add_gaussian_noise(sensor_data.signal, snr_db=20),
                'Baseline_Wander': add_baseline_wander(sensor_data.signal, sensor_data.fs),
                'Powerline_50Hz': add_powerline_interference(sensor_data.signal, sensor_data.fs),
                'Motion_Artifact': add_motion_artifact(sensor_data.signal, sensor_data.fs)
            }
            
            for cond_name, noisy_sig in noise_conditions.items():
                n_res = evaluate_condition(noisy_sig, sensor_data.fs, cond_name)
                results.extend(n_res)
                
            for target_fs in sampling_freqs:
                if sensor_data.fs <= target_fs:
                    continue
                num_samples = int(len(sensor_data.signal) * target_fs / sensor_data.fs)
                downsampled_sig = scipy.signal.resample(sensor_data.signal, num_samples)
                d_res = evaluate_condition(downsampled_sig, target_fs, f'Downsampled_{target_fs}Hz')
                results.extend(d_res)
                
        except Exception as e:
            print(f"Error processing {sensor_data.label}: {e}")
            
    return pd.DataFrame(results), pd.DataFrame(error_results)

def get_bidmc_generator():
    for i in range(3): 
        try:
            yield load_bidmc_data(subject_idx=i)
        except Exception as e: pass

def get_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def get_mimic_generator():
    base_dir = os.path.join(get_data_dir(), 'mimic3wdb_matched', 'p00')
    hea_files = glob(os.path.join(base_dir, '*/*_000*.hea'))
    for hea in hea_files[:3]: 
        record_path = hea.replace('.hea', '')
        try:
            data = load_mimic_data(record_path)
            if data is not None: yield data
        except Exception as e: pass

def get_wesad_generator():
    pkl_files = glob(os.path.join(get_data_dir(), 'wesad', 'WESAD', 'S*', 'S*.pkl'))
    for pkl in pkl_files[:3]:
        try:
            yield load_wesad_data(pkl)
        except Exception as e: pass

def get_dalia_generator():
    pkl_files = glob(os.path.join(get_data_dir(), 'ppg+dalia', 'ppg_dalia_extracted', 'PPG_FieldStudy', 'S*', 'S*.pkl'))
    for pkl in pkl_files[:3]:
        try:
            yield load_dalia_data(pkl)
        except Exception as e: pass

if __name__ == "__main__":
    all_results = pd.DataFrame()
    all_errors = pd.DataFrame()
    
    generators = {
        'BIDMC': get_bidmc_generator(),
        'WESAD': get_wesad_generator(),
        'PPG-DaLiA': get_dalia_generator(),
    }
    
    for name, gen in generators.items():
        df, err_df = benchmark_dataset(gen, name)
        all_results = pd.concat([all_results, df], ignore_index=True)
        all_errors = pd.concat([all_errors, err_df], ignore_index=True)
        
    print("\n\n=== BENCHMARK SUMMARY ===")
    all_results.to_csv('benchmark_results_comprehensive.csv', index=False)
    all_errors.to_csv('benchmark_errors.csv', index=False)
    print("Saved benchmark_results_comprehensive.csv and benchmark_errors.csv")
    
    save_reproducibility_log()
