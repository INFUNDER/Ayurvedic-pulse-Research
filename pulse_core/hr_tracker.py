"""
Stage 2 - multi-channel heart-rate tracking.

The rest of the pipeline adapts its filters to the subject's *own* heart rate,
so HR has to be estimated before any beat is detected. Approach (in the spirit
of TROIKA / JOSS spectral peak tracking, Zhang et al. 2015):

1. Short-time spectra (8 s windows, 1 s hop) of every optical channel.
2. Each window is winsorised at +-3 robust SDs so transients (force steps,
   motion spikes) cannot dominate; then harmonic summation: P(f) + 0.5 P(2f) + 0.33 P(3f), so a pulse whose energy
   sits in its harmonics (sharp upstroke, strong dicrotic notch) still votes for
   the right fundamental.
3. Channel fusion: the three sensors sit on the same radial artery, so the cardiac
   peak is common to all of them while most noise is not. Normalised spectra are
   averaged, weighted by each channel's spectral peakiness and valid-sample fraction.
4. Viterbi decoding over windows with a penalty on HR jumps, which gives a smooth
   physiologically plausible track instead of jumping to motion peaks.
"""
from dataclasses import dataclass

import numpy as np

from .preprocess import bandpass


@dataclass
class HRTrack:
    t: np.ndarray           # window centres, s
    hr_bpm: np.ndarray
    confidence: np.ndarray  # 0..1, share of fused spectral power near the chosen HR

    def at(self, t_s):
        """HR (bpm) interpolated at arbitrary times."""
        return np.interp(t_s, self.t, self.hr_bpm)

    def per_sample(self, n, fs):
        return self.at(np.arange(n) / fs)

    @property
    def median_hz(self):
        return float(np.median(self.hr_bpm)) / 60.0


def _harmonic_spectrum(seg, fs, freqs, clip_k=3.0):
    # winsorise at +-k robust SDs: force steps / motion spikes are broadband and
    # would otherwise flood the spectrum, while the periodic pulse is barely touched
    med = np.median(seg)
    sd = 1.4826 * np.median(np.abs(seg - med)) + 1e-12
    seg = np.clip(seg, med - clip_k * sd, med + clip_k * sd)
    seg = (seg - seg.mean()) * np.hanning(len(seg))
    nfft = int(2 ** np.ceil(np.log2(max(len(seg), fs * 60 / 0.5))))  # ~0.5 bpm bins
    spec = np.abs(np.fft.rfft(seg, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = lambda ff: np.interp(ff, f, spec, right=0.0)
    return p(freqs) + 0.5 * p(2 * freqs) + 0.33 * p(3 * freqs)


def track_heart_rate(signals, fs, valid=None, hr_range=(40, 180), win_s=8.0, step_s=1.0,
                     max_jump_bpm_per_s=8.0):
    """
    signals : (C, N) pulsatile signals (any scale; each channel is normalised)
    valid   : (C, N) bool, optional
    """
    signals = np.atleast_2d(signals)
    C, N = signals.shape
    valid = np.ones_like(signals, bool) if valid is None else np.atleast_2d(valid)
    x = bandpass(signals, fs, 0.5, min(8.0, 0.45 * fs))

    bpm_grid = np.arange(hr_range[0], hr_range[1] + 0.5, 0.5)
    freqs = bpm_grid / 60.0
    win = int(min(win_s, N / fs) * fs)
    step = int(step_s * fs)
    starts = np.arange(0, N - win + 1, step) if N > win else np.array([0])

    obs = np.zeros((len(starts), len(freqs)))
    for k, s in enumerate(starts):
        fused = np.zeros(len(freqs))
        for c in range(C):
            vfrac = valid[c, s:s + win].mean()
            if vfrac < 0.5:
                continue
            spec = _harmonic_spectrum(x[c, s:s + win], fs, freqs)
            tot = spec.sum()
            if tot <= 0:
                continue
            spec /= tot
            peakiness = spec.max() * len(spec)  # 1 = flat, large = one clear peak
            fused += spec * vfrac * np.log1p(peakiness)
        obs[k] = fused / fused.sum() if fused.sum() > 0 else 1.0 / len(freqs)

    # Viterbi: log-emission + Gaussian log-transition on HR change
    sigma = max_jump_bpm_per_s * step_s
    d = bpm_grid[:, None] - bpm_grid[None, :]
    log_trans = -0.5 * (d / sigma) ** 2
    log_obs = np.log(obs + 1e-12)
    score = log_obs[0].copy()
    back = np.zeros(obs.shape, int)
    for k in range(1, len(starts)):
        cand = score[:, None] + log_trans  # prev x next
        back[k] = np.argmax(cand, axis=0)
        score = cand[back[k], np.arange(len(freqs))] + log_obs[k]
    path = np.zeros(len(starts), int)
    path[-1] = int(np.argmax(score))
    for k in range(len(starts) - 1, 0, -1):
        path[k - 1] = back[k, path[k]]

    hr = bpm_grid[path]
    band = np.abs(bpm_grid[None, :] - hr[:, None]) <= 4.0
    conf = (obs * band).sum(axis=1)
    centres = (starts + win / 2) / fs
    return HRTrack(centres, hr, conf)
