import numpy as np
import scipy.signal
from pulse_core.sqi import calculate_sqi, evaluate_signal_quality
from pulse_core.core_types import DetectionResult, BaseDetector

class ClinicalDetector(BaseDetector):
    """
    Detector A: High-precision detector optimized for clean, clinical data.
    Uses Savitzky-Golay smoothing and adaptive Z-score amplitude thresholding.
    """
    def detect(self, signal: np.ndarray, fs: float, sqi_score: float = 0.0) -> DetectionResult:
        vpg = np.gradient(signal)
        apg = np.gradient(vpg)
        
        # Smooth derivatives
        window = int(0.05*fs) | 1
        poly = min(3, window - 1) if window > 1 else 0
        if window > 2:
            vpg = scipy.signal.savgol_filter(vpg, window_length=window, polyorder=poly)
            apg = scipy.signal.savgol_filter(apg, window_length=window, polyorder=poly)
        
        # Detect Systolic Peaks
        min_dist = int(0.3 * fs)
        # Dynamic Amplitude Scaling: instead of hardcoded 0.5, use 50% of the signal's standard deviation
        dynamic_prominence = max(0.01, np.std(signal) * 0.5)
        coarse_peaks, _ = scipy.signal.find_peaks(signal, distance=min_dist, prominence=dynamic_prominence)
        
        # Detect Onsets
        onsets = []
        for i in range(len(coarse_peaks)):
            search_start = max(0, coarse_peaks[i] - int(0.4 * fs))
            segment = signal[search_start:coarse_peaks[i]]
            if len(segment) > 0:
                onset_idx = search_start + np.argmin(segment)
                onsets.append(onset_idx)
        
        return DetectionResult(
            systolic_peaks=coarse_peaks,
            onsets=np.array(onsets),
            confidence=1.0,
            detector_used='clinical',
            morphology_available=False,  # Clinical detector skips a/b/c/d/e fiducials for speed/robustness
            sqi_score=sqi_score,
            rejection_reason="None",
            fiducials={'vpg': vpg, 'apg': apg}
        )

class WearableDetector(BaseDetector):
    """
    Detector B: Wearable-optimized detector.
    Uses HR-adaptive dual moving averages, morphological derivative checks, 
    and local SNR rejection to handle extreme amplitude variations.
    """
    def detect(self, signal: np.ndarray, fs: float, sqi_score: float = 0.0) -> DetectionResult:
        # 1. Estimate HR to set adaptive windows
        freqs, psd = scipy.signal.welch(signal, fs, nperseg=min(2048, len(signal)))
        hr_band = (freqs >= 0.8) & (freqs <= 3.0) 
        
        if np.sum(hr_band) == 0 or np.sum(psd[hr_band]) == 0:
            est_hr_hz = 1.2 
        else:
            est_hr_hz = freqs[hr_band][np.argmax(psd[hr_band])]
            
        beat_interval_sec = 1.0 / est_hr_hz
        
        w1_sec = 0.12 * beat_interval_sec
        w2_sec = 1.0 * beat_interval_sec
        
        w1_samples = max(3, int(w1_sec * fs))
        w2_samples = max(w1_samples * 2, int(w2_sec * fs))
        
        sq_signal = signal ** 2
        
        ma_peak = np.convolve(sq_signal, np.ones(w1_samples)/w1_samples, mode='same')
        ma_beat = np.convolve(sq_signal, np.ones(w2_samples)/w2_samples, mode='same')
        
        alpha = 0.02 * np.mean(sq_signal)
        roi = ma_peak > (ma_beat + alpha)
        
        blocks = []
        in_block = False
        start_idx = 0
        for i, val in enumerate(roi):
            if val and not in_block:
                in_block = True
                start_idx = i
            elif not val and in_block:
                in_block = False
                blocks.append((start_idx, i))
        if in_block:
            blocks.append((start_idx, len(roi)))
            
        deriv = np.gradient(signal)
        raw_peaks = []
        min_dist_samples = int(0.3 * fs) 
        
        for b_start, b_end in blocks:
            if b_end - b_start < 2:
                continue
                
            block_sig = signal[b_start:b_end]
            local_max_idx = b_start + np.argmax(block_sig)
            
            if len(raw_peaks) > 0 and (local_max_idx - raw_peaks[-1]) < min_dist_samples:
                if signal[local_max_idx] > signal[raw_peaks[-1]]:
                    raw_peaks[-1] = local_max_idx
                continue
                
            search_start = max(b_start, local_max_idx - 2)
            search_end = min(b_end, local_max_idx + 3)
            d_window = deriv[search_start:search_end]
            
            if len(d_window) > 1 and np.max(d_window) > 0 and np.min(d_window) < 0:
                raw_peaks.append(local_max_idx)
                
        # 2. Local SNR rejection
        accepted_peaks = []
        snr_window_samples = int(1.0 * fs) # 1-second window for local noise estimation
        snr_threshold = 1.0 # Peak must be at least 1 RMS unit above local mean
        
        for p in raw_peaks:
            win_start = max(0, p - snr_window_samples // 2)
            win_end = min(len(signal), p + snr_window_samples // 2)
            local_win = signal[win_start:win_end]
            
            if len(local_win) > 0:
                local_mean = np.mean(local_win)
                local_rms = np.sqrt(np.mean((local_win - local_mean)**2))
                
                if local_rms > 0:
                    local_snr = (signal[p] - local_mean) / local_rms
                    if local_snr >= snr_threshold:
                        accepted_peaks.append(p)
                else:
                    accepted_peaks.append(p) # Flat signal logic edge case
                    
        accepted_peaks = np.array(accepted_peaks)
        
        confidence = 0.8
        if len(accepted_peaks) > 0:
            signal_power = np.mean(sq_signal[accepted_peaks]) if len(accepted_peaks) > 0 else 0
            noise_power = np.mean(sq_signal)
            if noise_power > 0 and signal_power / noise_power > 2.0:
                confidence = 0.85
                
        return DetectionResult(
            systolic_peaks=accepted_peaks,
            onsets=np.array([]), # Skipping onsets in wearable for simplicity, can be added later
            confidence=confidence,
            detector_used='wearable',
            morphology_available=False,
            sqi_score=sqi_score,
            rejection_reason="None" if len(accepted_peaks) > 0 else "Low Local SNR / No periodicity",
            fiducials={'vpg': deriv, 'apg': np.gradient(deriv)}
        )

def detect_beats(signal, fs, mode="auto", sqi_class=None, sqi_score=0.0):
    """
    Detects fiducial points dynamically based on signal quality.
    modes: 'auto', 'clinical', 'wearable'
    Returns: DetectionResult
    """
    clinical = ClinicalDetector()
    wearable = WearableDetector()
    
    if mode == "clinical":
        return clinical.detect(signal, fs, sqi_score)
    elif mode == "wearable":
        return wearable.detect(signal, fs, sqi_score)
    elif mode == "auto":
        if sqi_class is None:
            sqi_metrics = calculate_sqi(signal, fs)
            sqi_score = sqi_metrics['Composite_SQI']
            sqi_class = evaluate_signal_quality(sqi_score)
            
        if sqi_class == 'Good':
            return clinical.detect(signal, fs, sqi_score)
        elif sqi_class == 'Moderate':
            return wearable.detect(signal, fs, sqi_score)
        else: # Poor SQI
            return DetectionResult(
                confidence=0.0,
                detector_used='rejected',
                sqi_score=sqi_score,
                rejection_reason="Poor SQI"
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")
