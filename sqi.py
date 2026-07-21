import numpy as np
import scipy.signal
from scipy.stats import skew, kurtosis
from config import SQI_GOOD_THRESHOLD, SQI_MODERATE_THRESHOLD

def _calculate_acf_peak(signal, fs):
    """
    Calculates the normalized autocorrelation peak height as a proxy for 
    signal periodicity and waveform self-consistency without requiring prior beat detection.
    """
    if len(signal) == 0:
        return 0.0
        
    # Standardize signal to compute normalized ACF
    sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    
    # Compute ACF via FFT (Wiener-Khinchin theorem)
    fft_val = np.fft.rfft(sig_norm)
    acf = np.fft.irfft(fft_val * np.conj(fft_val))
    
    # Normalize ACF so that zero-lag is 1.0
    if acf[0] == 0:
        return 0.0
    acf = acf / acf[0]
    
    # Look for the first major peak in the physiological HR range (40 - 180 BPM)
    min_lag = int(fs * (60.0 / 180.0))
    max_lag = int(fs * (60.0 / 40.0))
    
    if max_lag >= len(acf):
        max_lag = len(acf) - 1
        
    if min_lag >= max_lag:
        return 0.0
        
    search_window = acf[min_lag:max_lag]
    peaks, _ = scipy.signal.find_peaks(search_window)
    
    if len(peaks) > 0:
        # Return the height of the most prominent peak in the search window
        return np.max(search_window[peaks])
    else:
        # Fallback to absolute max if no distinct peaks found
        return np.max(search_window) if len(search_window) > 0 else 0.0


def calculate_sqi(signal, fs):
    """
    Calculates Signal Quality Indices (SQI) for a given PPG signal segment.
    Returns individual metrics and a Composite SQI.
    """
    # 1. Spectral Power Ratio (SPR)
    freqs, psd = scipy.signal.welch(signal, fs, nperseg=min(4096, len(signal)))
    hr_band = (freqs >= 0.5) & (freqs <= 3.0)
    
    total_power = np.sum(psd)
    spr = np.sum(psd[hr_band]) / total_power if total_power > 0 else 0.0
        
    # 2. Kurtosis (waveform morphology)
    # scipy.stats.kurtosis defaults to Fisher (excess kurtosis, normal = 0).
    kurt = kurtosis(signal)
    
    # 3. Autocorrelation (periodicity)
    acf_peak = _calculate_acf_peak(signal, fs)
    
    # 4. Composite SQI Calculation
    # Normalize SPR (0 to 1) -> Already 0 to 1
    norm_spr = np.clip(spr, 0, 1)
    
    # Normalize Kurtosis (0 to 1)
    # Clean PPG typically has excess kurtosis around 1 to 5. 
    # Heavily corrupted signals (or flat signals) often approach 0 or become very large (spike noise).
    # We'll map kurtosis into a [0, 1] confidence range. Values between 1 and 10 are ideal.
    norm_kurt = np.clip(kurt / 10.0, 0, 1) if kurt > 0 else 0.0
    
    # Normalize ACF (0 to 1)
    norm_acf = np.clip(acf_peak, 0, 1)
    
    # Unweighted mean for composite
    composite_sqi = float(np.mean([norm_spr, norm_kurt, norm_acf]))
    
    return {
        'Composite_SQI': composite_sqi,
        'SPR': spr,
        'Kurtosis': kurt,
        'ACF_Peak': acf_peak
    }

def evaluate_signal_quality(composite_sqi):
    """
    Classifies signal quality based on the Composite SQI.
    Returns 'Good', 'Moderate', or 'Poor'.
    """
    if composite_sqi >= SQI_GOOD_THRESHOLD:
        return 'Good'
    elif composite_sqi >= SQI_MODERATE_THRESHOLD:
        return 'Moderate'
    else:
        return 'Poor'
