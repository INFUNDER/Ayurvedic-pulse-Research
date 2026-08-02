import numpy as np
from feature_extraction import extract_all_features
from beat_detection import WearableDetector
from sqi import calculate_sqi, evaluate_signal_quality
import warnings
from scipy.signal import butter, sosfiltfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data)

class DoshaCalibrator:
    def __init__(self, fs, duration_sec=60):
        self.fs = fs
        self.duration_sec = duration_sec
        self.buffer = []
        self.target_samples = int(fs * duration_sec)
        
    def add_data(self, data):
        self.buffer.extend(data)
        
    def is_ready(self):
        return len(self.buffer) >= self.target_samples
        
    def calibrate(self):
        """
        Runs the calibration routine on the collected buffer.
        Returns a dictionary containing the baseline median and std for h1 and AI.
        """
        if not self.is_ready():
            return None
            
        raw_signal = np.array(self.buffer[:self.target_samples])
        
        # Pre-filter the entire buffer to avoid edge artifacts in windows
        signal = bandpass_filter(raw_signal, 0.5, 10.0, self.fs)
        
        # Segment into 5-second windows to filter out local artifacts and compute features
        window_size = int(5 * self.fs)
        
        valid_h1 = []
        valid_ai = []
        
        detector = WearableDetector()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i in range(0, len(signal), window_size):
                window = signal[i:i+window_size]
                if len(window) < window_size // 2:
                    break
                    
                # 1. Artifact Rejection using SQI
                sqi_metrics = calculate_sqi(window, self.fs)
                sqi_class = evaluate_signal_quality(sqi_metrics['Composite_SQI'])
                
                # We drop Poor quality windows to prevent motion artifacts from skewing baseline
                if sqi_class == 'Poor':
                    continue
                    
                # 2. Extract features
                det_result = detector.detect(window, self.fs)
                if len(det_result.systolic_peaks) >= 3:
                    det_result.confidence = 0.90 if sqi_class == 'Good' else 0.70
                    
                    features, status = extract_all_features(window, self.fs, det_result)
                    
                    if status == "SUCCESS":
                        if 'h1_Amplitude_P' in features and not np.isnan(features['h1_Amplitude_P']):
                            valid_h1.append(features['h1_Amplitude_P'])
                        if 'Augmentation_Index' in features and not np.isnan(features['Augmentation_Index']):
                            valid_ai.append(features['Augmentation_Index'])
                            
        if len(valid_h1) < 2 or len(valid_ai) < 2:
            return None # Not enough clean data to establish a baseline
            
        baseline_profile = {
            'h1_median': float(np.nanmedian(valid_h1)),
            'h1_std': float(np.nanstd(valid_h1)),
            'ai_median': float(np.nanmedian(valid_ai)),
            'ai_std': float(np.nanstd(valid_ai))
        }
        
        return baseline_profile
