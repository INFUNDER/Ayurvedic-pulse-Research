"""
Stage 3 - HR-tracking filtering and data-driven wavelet shrinkage.

A fixed 0.5-10 Hz band-pass treats a 48 bpm and a 110 bpm person the same way.
Here the band edges move with the tracked heart rate f0(t):

    high-pass edge = hp_factor x f0        (removes respiration/drift below the beat)
    low-pass edge  = n_harmonics x f0      (keeps the harmonics that carry the
                                            dicrotic notch, drops everything above)

The signal is processed in overlapping Hann blocks, each with its own zero-phase
filter, and recombined by overlap-add, so the filter is time-varying but never
shifts the waveform in time (fiducial timing is preserved).

Wavelet stage: BayesShrink (Chang, Yu & Vetterli 2000). The noise level is
estimated from the finest detail coefficients with the median absolute deviation
(sigma = MAD / 0.6745, Donoho & Johnstone 1994) and the threshold is set
separately per level. Nothing is hand-tuned, and the threshold scales with
each recording's own noise.
"""
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from .preprocess import _ola_windows


def _sos(kind, fc, fs, order):
    return butter(order, min(fc / (0.5 * fs), 0.99), btype=kind, output="sos")


def tracking_filter(x, fs, hr_track, kind, factor, block_s=4.0, pad_s=3.0, order=3,
                    min_hz=0.3, max_hz=15.0):
    """
    Time-varying zero-phase high- or low-pass whose cutoff = factor x HR(t).
    kind: 'high' or 'low'.
    """
    n = len(x)
    win, starts = _ola_windows(n, fs, block_s, block_s / 2)
    w = np.hanning(win)
    pad = int(pad_s * fs)
    out = np.zeros(n)
    wsum = np.zeros(n)
    for s in starts:
        centre_t = (s + win / 2) / fs
        fc = np.clip(factor * hr_track.at(centre_t) / 60.0, min_hz, min(max_hz, 0.45 * fs))
        a, b = max(0, s - pad), min(n, s + win + pad)
        y = sosfiltfilt(_sos(kind, fc, fs, order), x[a:b])
        out[s:s + win] += y[s - a:s - a + win] * w
        wsum[s:s + win] += w
    return out / np.maximum(wsum, 1e-9)


def wavelet_bayes_shrink(x, wavelet="sym6", level=None):
    """Soft-threshold BayesShrink denoising with noise estimated from the data itself."""
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    level = min(level or 5, max_level)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma_n = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma_n <= 0:
        return x.copy(), 0.0
    out = [coeffs[0]]
    for d in coeffs[1:]:
        sigma_x = np.sqrt(max(np.mean(d**2) - sigma_n**2, 1e-12))
        thr = sigma_n**2 / sigma_x
        out.append(pywt.threshold(d, thr, mode="soft"))
    y = pywt.waverec(out, wavelet)[: len(x)]
    return y, float(sigma_n)


def adaptive_denoise(x, fs, hr_track, hp_factor=0.5, n_harmonics=8, use_wavelet=True,
                     max_hz=15.0):
    """High-pass(f0) -> wavelet shrinkage -> low-pass(n*f0). Returns (clean, noise_sigma)."""
    y = tracking_filter(x, fs, hr_track, "high", hp_factor)
    sigma = 0.0
    if use_wavelet:
        y, sigma = wavelet_bayes_shrink(y)
    y = tracking_filter(y, fs, hr_track, "low", n_harmonics, max_hz=max_hz)
    return y, sigma


def fixed_bandpass(x, fs, lo=0.5, hi=10.0, order=4):
    """The v1 approach, kept as the baseline for benchmarking."""
    sos = butter(order, [lo / (0.5 * fs), min(hi / (0.5 * fs), 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, x)
