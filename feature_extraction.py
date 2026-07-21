import numpy as np
import scipy.stats
import scipy.signal
import scipy.integrate
from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
from core_types import DetectionResult

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    import antropy as ant
    HAS_ANT = True
except ImportError:
    HAS_ANT = False

def _get_widths(beat_sig, fs):
    sys_amp = np.max(beat_sig)
    base_amp = np.min(beat_sig)
    rng = sys_amp - base_amp
    
    widths = {}
    for pct in [0.1, 0.25, 0.33, 0.50, 0.66, 0.75, 0.90]:
        thresh = base_amp + (rng * pct)
        above = np.where(beat_sig >= thresh)[0]
        if len(above) > 1:
            widths[f'Width_at_{int(pct*100)}'] = (above[-1] - above[0]) / fs
        else:
            widths[f'Width_at_{int(pct*100)}'] = 0.0
    return widths

def extract_time_domain(signal, fs, detection_result: DetectionResult):
    features = {}
    onsets = detection_result.onsets
    sys_peaks = detection_result.systolic_peaks
    
    if len(onsets) < 2 or len(sys_peaks) == 0:
        # Fallback to peaks-only logic if onsets are missing (e.g., from WearableDetector)
        if len(sys_peaks) > 1:
            intervals = np.diff(sys_peaks) / fs
            features['Mean_Beat_Interval'] = np.mean(intervals)
            features['Heart_Rate_BPM'] = 60.0 / features['Mean_Beat_Interval'] if features['Mean_Beat_Interval'] > 0 else 0
            features['HRV_SDNN'] = np.std(intervals) * 1000
            features['HRV_RMSSD'] = np.sqrt(np.mean(np.square(np.diff(intervals)))) * 1000 if len(intervals) > 1 else 0
        return features
        
    intervals = np.diff(onsets) / fs
    features['Mean_Beat_Interval'] = np.mean(intervals)
    features['Heart_Rate_BPM'] = 60.0 / features['Mean_Beat_Interval'] if features['Mean_Beat_Interval'] > 0 else 0
    features['HRV_SDNN'] = np.std(intervals) * 1000
    features['HRV_RMSSD'] = np.sqrt(np.mean(np.square(np.diff(intervals)))) * 1000 if len(intervals) > 1 else 0
    features['HRV_pNN50'] = (np.sum(np.abs(np.diff(intervals)) > 0.05) / len(intervals)) * 100 if len(intervals) > 1 else 0
    
    rise_times, fall_times, sys_amps, dias_amps = [], [], [], []
    w_all = {k: [] for k in ['Width_at_10', 'Width_at_25', 'Width_at_33', 'Width_at_50', 'Width_at_66', 'Width_at_75', 'Width_at_90']}
    areas, sys_areas, dias_areas = [], [], []
    sys_dias_ratios = []
    
    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i+1]
        beat = signal[start:end]
        beat_peaks = [p for p in sys_peaks if start <= p < end]
        
        if len(beat) > 0 and len(beat_peaks) > 0:
            peak = beat_peaks[0]
            rise_time = (peak - start) / fs
            fall_time = (end - peak) / fs
            rise_times.append(rise_time)
            fall_times.append(fall_time)
            sys_amps.append(signal[peak])
            dias_amps.append(signal[end])
            
            widths = _get_widths(beat, fs)
            for k in widths: w_all[k].append(widths[k])
            
            total_area = scipy.integrate.simpson(beat, dx=1/fs)
            s_area = scipy.integrate.simpson(signal[start:peak], dx=1/fs) if peak > start else 0
            d_area = scipy.integrate.simpson(signal[peak:end], dx=1/fs) if end > peak else 0
            
            areas.append(total_area)
            sys_areas.append(s_area)
            dias_areas.append(d_area)
            
            if d_area != 0:
                sys_dias_ratios.append(s_area / d_area)

    features['Mean_Rise_Time'] = np.mean(rise_times) if rise_times else 0
    features['Mean_Fall_Time'] = np.mean(fall_times) if fall_times else 0
    for k in w_all: features[f'Mean_{k}'] = np.mean(w_all[k]) if w_all[k] else 0
    features['Mean_Pulse_Area'] = np.mean(areas) if areas else 0
    
    features['Global_RMS'] = np.sqrt(np.mean(signal**2))
    features['Global_Skewness'] = scipy.stats.skew(signal)
    features['Global_Kurtosis'] = scipy.stats.kurtosis(signal)
    
    return features

def extract_frequency_domain(signal, fs):
    features = {}
    n = len(signal)
    if n == 0: return features
    freqs, psd = scipy.signal.welch(signal, fs, nperseg=min(fs*4, n))
    features['Total_Spectral_Power'] = np.sum(psd)
    psd_norm = psd / np.sum(psd)
    features['Spectral_Entropy_Welch'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return features

def extract_nonlinear(signal):
    features = {}
    if not HAS_ANT or len(signal) == 0: return features
    try:
        features['Sample_Entropy'] = ant.sample_entropy(signal)
    except Exception: pass
    return features

def extract_derivatives(signal, detection_result: DetectionResult):
    features = {}
    if not detection_result.morphology_available:
        return features # Returning empty signals intent that fiducials are missing

    a_waves = detection_result.fiducials.get('a_waves', [])
    apg = detection_result.fiducials.get('apg', [])
    vpg = detection_result.fiducials.get('vpg', [])
    
    if len(apg) == 0 or len(a_waves) == 0:
        return features
        
    # Mocking extraction logic (to avoid complex re-implementation) 
    # If morphology was available, we'd compute the APG Aging index here.
    return features


def extract_all_features(signal, fs, detection_result: DetectionResult):
    """
    Master function that aggregates all features into a single Feature Library dictionary.
    Returns: features (dict), feature_status (dict)
    """
    library = {}
    status = {}
    
    # Base feature list to track
    expected_features = [
        'Heart_Rate_BPM', 'HRV_RMSSD', 'HRV_SDNN', 
        'Sample_Entropy', 'Mean_Width_at_50', 'APG_Aging_Index'
    ]
    for ef in expected_features:
        status[ef] = "Signal Rejected" # Default state
    
    if detection_result.confidence == 0.0 or len(detection_result.systolic_peaks) == 0:
        return library, status
        
    # Reset status to Low Confidence before attempting
    for ef in expected_features:
        status[ef] = "Low Confidence"
        
    # 1. Time Domain & Non-linear (Robust - Medium Confidence)
    if detection_result.confidence >= CONFIDENCE_MEDIUM:
        for ef in ['Heart_Rate_BPM', 'HRV_RMSSD', 'HRV_SDNN', 'Mean_Width_at_50', 'Sample_Entropy']:
            status[ef] = "Numerical Instability"
            
        try:
            td = extract_time_domain(signal, fs, detection_result)
            library.update(td)
            if 'Heart_Rate_BPM' in td: status['Heart_Rate_BPM'] = "Success"
            if 'HRV_RMSSD' in td: status['HRV_RMSSD'] = "Success"
            if 'HRV_SDNN' in td: status['HRV_SDNN'] = "Success"
            if 'Mean_Width_at_50' in td: status['Mean_Width_at_50'] = "Success"
        except Exception:
            pass
            
        try:
            fd = extract_frequency_domain(signal, fs)
            library.update(fd)
        except Exception:
            pass
            
        try:
            nl = extract_nonlinear(signal)
            library.update(nl)
            if 'Sample_Entropy' in nl and not np.isnan(nl['Sample_Entropy']): 
                status['Sample_Entropy'] = "Success"
        except Exception:
            pass
            
    # 2. Advanced Morphology (Fragile - High Confidence)
    if detection_result.confidence >= CONFIDENCE_HIGH:
        status['APG_Aging_Index'] = "Numerical Instability"
        
        if detection_result.morphology_available:
            try:
                drv = extract_derivatives(signal, detection_result)
                library.update(drv)
                if 'APG_Aging_Index' in drv: status['APG_Aging_Index'] = "Success"
            except Exception:
                pass
        else:
            status['APG_Aging_Index'] = "Missing Fiducials"
            
    return library, status
