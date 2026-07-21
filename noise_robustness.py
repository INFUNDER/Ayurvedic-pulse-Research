import numpy as np

def add_gaussian_noise(signal, snr_db):
    """
    Adds Gaussian white noise to achieve a specific Signal-to-Noise Ratio (SNR).
    """
    if len(signal) == 0:
        return signal
        
    # Ensure signal is zero mean for power calculation
    sig_centered = signal - np.mean(signal)
    sig_power = np.mean(sig_centered**2)
    
    if sig_power == 0:
        return signal
        
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

def add_baseline_wander(signal, fs, bw_freq=0.3, amplitude_ratio=0.5):
    """
    Adds a low frequency baseline wander.
    amplitude_ratio defines peak-to-peak amplitude relative to signal's range.
    """
    t = np.arange(len(signal)) / fs
    bw = np.sin(2 * np.pi * bw_freq * t)
    sig_range = np.max(signal) - np.min(signal)
    bw = bw * (sig_range * amplitude_ratio * 0.5)
    return signal + bw

def add_powerline_interference(signal, fs, pl_freq=50.0, amplitude_ratio=0.1):
    """
    Adds 50Hz or 60Hz powerline interference.
    """
    t = np.arange(len(signal)) / fs
    pl = np.sin(2 * np.pi * pl_freq * t)
    sig_range = np.max(signal) - np.min(signal)
    pl = pl * (sig_range * amplitude_ratio * 0.5)
    return signal + pl

def add_motion_artifact(signal, fs, num_artifacts=3, max_duration_sec=2.0, amplitude_ratio=2.0):
    """
    Simulates motion artifacts as transient high-amplitude disturbances.
    """
    if len(signal) == 0:
        return signal
        
    noisy_sig = np.copy(signal)
    n = len(signal)
    
    for _ in range(num_artifacts):
        start = np.random.randint(0, max(1, n - int(fs)))
        duration = np.random.randint(int(fs * 0.5), max(int(fs * 0.5) + 1, int(fs * max_duration_sec)))
        end = min(start + duration, n)
        
        if end <= start:
            continue
            
        t = np.arange(end - start) / fs
        # Random motion freq between 1 and 3 Hz
        freq = np.random.uniform(1.0, 3.0)
        
        sig_range = np.max(signal) - np.min(signal)
        artifact = np.sin(2 * np.pi * freq * t) * (sig_range * amplitude_ratio)
        
        # Apply hanning window to smooth edges
        window = np.hanning(len(artifact))
        artifact *= window
        
        noisy_sig[start:end] += artifact
        
    return noisy_sig
