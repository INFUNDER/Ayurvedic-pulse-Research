import numpy as np
import scipy.signal

def remove_noise(signal, fs, hp_hz=0.5, lp_hz=20.0, trim_sec=0.5, use_hp=True, use_lp=True):
    """
    Filters the PPG signal using a Butterworth bandpass approach.
    """
    nyq = 0.5 * fs
    filtered = np.copy(signal)
    
    # High-pass filter
    if use_hp:
        b_hp, a_hp = scipy.signal.butter(4, hp_hz / nyq, btype='high')
        filtered = scipy.signal.filtfilt(b_hp, a_hp, filtered)
    
    # Low-pass filter
    if use_lp:
        b_lp, a_lp = scipy.signal.butter(4, min(lp_hz / nyq, 0.99), btype='low')
        filtered = scipy.signal.filtfilt(b_lp, a_lp, filtered)
    
    # Standardize signal (zero mean, unit variance)
    filtered = (filtered - np.mean(filtered)) / np.std(filtered)
    
    trim_samples = int(trim_sec * fs)
    return filtered, trim_samples
