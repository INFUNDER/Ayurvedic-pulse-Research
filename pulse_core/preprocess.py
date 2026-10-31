"""
Stage 1 - subject-independent normalisation and contact-force cancellation.

Why AC/DC normalisation?
    Raw MAX30102 counts depend on skin tone, tissue depth, LED current and how
    hard the sensor is pressed. Dividing the pulsatile (AC) part by the local
    DC level gives a dimensionless signal (the perfusion signal, in %) that is
    comparable between people and sessions. This removes the main reason fixed
    amplitude thresholds fail.

Why the FSR is filtered before it is used as a reference
    Each FSR402 sits behind its MAX30102, so it measures that sensor's hold-down
    force. Changes in force move the optical baseline and gain. But an FSR on the
    radial artery also picks up the arterial *pressure pulse*. If that pulse is
    left in the reference, an adaptive canceller will happily subtract the PPG
    pulse itself. So the reference is low-passed below the subject's own heart
    rate (cutoff = `ref_hr_fraction` x HR) before regression.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass(x, fs, fc, order=2):
    sos = butter(order, min(fc / (0.5 * fs), 0.99), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def bandpass(x, fs, lo, hi, order=3):
    sos = butter(order, [lo / (0.5 * fs), min(hi / (0.5 * fs), 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def to_pulsatile(ir, fs, dc_cutoff_hz=0.2):
    """
    Raw optical counts -> inverted AC/DC perfusion signal in percent.

    Blood absorbs IR light, so raw counts *fall* during systole; the sign flip
    makes systolic peaks positive. Returns (pulsatile, dc).
    """
    dc = lowpass(ir, fs, dc_cutoff_hz)
    dc = np.where(np.abs(dc) < 1e-9, 1e-9, dc)
    return -100.0 * (ir - dc) / dc, dc


def _ola_windows(n, fs, win_s, step_s):
    """Hann windows with 50 % overlap that sum to ~1 (used for blockwise adaptation)."""
    win = max(int(win_s * fs), 8)
    step = max(int(step_s * fs), 1)
    starts = list(range(0, max(n - win, 0) + 1, step))
    if starts[-1] + win < n:
        starts.append(n - win)
    return win, starts


def cancel_contact_force(ppg, fsr, fs, hr_hz, ref_hr_fraction=0.6, win_s=8.0):
    """
    Adaptive reference cancellation of contact-force artifacts.

    Blockwise least squares regression of the PPG onto the low-frequency FSR
    force (plus offset) in overlapping 8 s windows. The coupling coefficient is
    re-estimated in every window, so the canceller follows changes in strap
    tension, posture and sensor coupling instead of using one fixed gain.

    Design notes from the benchmark (scripts/benchmark_denoising.py):
      * a 4th-order low-pass on the reference is needed: a 2nd-order one leaks
        the arterial pressure pulse into the reference;
      * adding dF/dt as a regressor *hurts* (the derivative amplifies that leak
        and the canceller starts removing the pulse), so it is not used;
      * the coefficients are fitted against the low-passed PPG, not the raw PPG.
        Otherwise, with a steady strap, the reference is mostly leaked pulse and
        least squares uses it to subtract the PPG pulse (tests/test_pipeline.py).

    ppg, fsr : 1-D arrays (same length)
    hr_hz    : scalar or per-sample heart rate estimate in Hz
    Returns (cleaned_ppg, estimated_artifact).
    """
    n = len(ppg)
    fc = max(0.25, ref_hr_fraction * float(np.nanmin(np.atleast_1d(hr_hz))))
    f_lf = lowpass(fsr - np.mean(fsr), fs, fc, order=4)
    scale = np.std(f_lf)
    if scale < 1e-9:  # FSR flat: nothing to cancel
        return ppg.copy(), np.zeros(n)
    ref = np.vstack([f_lf / scale, np.ones(n)]).T
    # fit on the equally low-passed PPG: the coupling is then estimated from
    # sub-HR content only and can never be "explained" by the pulse itself
    ppg_lf = lowpass(ppg, fs, fc, order=4)

    win, starts = _ola_windows(n, fs, win_s, win_s / 2)
    w = np.hanning(win)
    artifact = np.zeros(n)
    wsum = np.zeros(n)
    for s in starts:
        seg = slice(s, s + win)
        coef, *_ = np.linalg.lstsq(ref[seg] * w[:, None], ppg_lf[seg] * w, rcond=None)
        artifact[seg] += (ref[seg] @ coef) * w
        wsum[seg] += w
    artifact /= np.maximum(wsum, 1e-9)
    return ppg - artifact, artifact
