"""
Contact-pressure calibration: subject-specific hold-down force for each sensor.

Physics: as external pressure on an artery rises, the pulsatile (AC) amplitude
first grows, peaks when transmural pressure ~ 0 (external ~ mean arterial
pressure), then falls as the vessel is occluded. This is the oscillometric
principle behind cuff BP monitors and is well documented for PPG
(e.g. Teng & Zhang 2004; review in Sensors 2020, PMC7570982).

Protocol: during the first ~60 s the user slowly tightens the strap (or presses
each sensor) from light to firm. Binning beats by FSR force and taking the median
amplitude per bin traces each person's own amplitude-vs-force curve, which gives:

  F_opt   - force with maximal pulse amplitude (person- and site-specific)
  F_light - rising-side half-maximum   (superficial palpation, classically Vata)
  F_deep  - falling-side half-maximum  (deep palpation, classically Kapha)
  width   - (F_deep - F_light)/F_opt, which tracks arterial compliance; a candidate
            objective proxy for Kathinya (vessel hardness)

This replaces fixed "Sparsha" ADC windows (100-350 / 400-700 / 750-1000) with
targets measured on the person. The light/medium/deep mapping to V/P/K is a
hypothesis to validate with a practitioner, not an established fact.
"""
import numpy as np
from scipy.signal import savgol_filter


def pressure_sweep_calibration(amplitude, force, n_bins=12, min_beats_per_bin=3):
    """
    amplitude : per-beat pulse amplitude (accepted beats, AC/DC units)
    force     : per-beat mean FSR reading for the same beats
    """
    amplitude, force = np.asarray(amplitude, float), np.asarray(force, float)
    ok = np.isfinite(amplitude) & np.isfinite(force)
    amplitude, force = amplitude[ok], force[ok]
    if len(force) < n_bins * min_beats_per_bin or np.ptp(force) <= 0:
        return None
    edges = np.unique(np.quantile(force, np.linspace(0, 1, n_bins + 1)))
    centres, amps = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (force >= lo) & (force <= hi)
        if m.sum() >= min_beats_per_bin:
            centres.append(np.median(force[m]))
            amps.append(np.median(amplitude[m]))
    centres, amps = np.array(centres), np.array(amps)
    if len(amps) < 5:
        return None
    smooth = savgol_filter(amps, 5 if len(amps) >= 5 else len(amps) | 1, 2)
    k = int(np.argmax(smooth))
    half = smooth[k] / 2.0
    rising = np.flatnonzero(smooth[:k] <= half)
    falling = np.flatnonzero(smooth[k:] <= half)
    f_light = np.interp(half, [smooth[rising[-1]], smooth[rising[-1] + 1]],
                        [centres[rising[-1]], centres[rising[-1] + 1]]) if len(rising) else np.nan
    f_deep = (np.interp(half, [smooth[k + falling[0]], smooth[k + falling[0] - 1]],
                        [centres[k + falling[0]], centres[k + falling[0] - 1]])
              if len(falling) else np.nan)
    f_opt = float(centres[k])
    return {
        "force_bins": centres, "amplitude": amps, "amplitude_smooth": smooth,
        "F_opt": f_opt, "F_light": float(f_light), "F_deep": float(f_deep),
        "width_rel": float((f_deep - f_light) / f_opt) if f_opt else np.nan,
        "peak_inside_sweep": 0 < k < len(smooth) - 1,
    }


def fit_fsr_calibration(adc_counts, grams):
    """
    FSR402 conductance is roughly proportional to force, so fit
    grams = a * counts^b on a few known weights (e.g. 50, 100, 200, 500, 1000 g)
    and return a callable counts -> newtons.
    """
    c, g = np.asarray(adc_counts, float), np.asarray(grams, float)
    b, log_a = np.polyfit(np.log(c), np.log(g), 1)
    a = np.exp(log_a)
    return lambda counts: a * np.power(np.maximum(counts, 1e-9), b) * 9.81e-3
