"""
Stage 6 - features from the subject's ensemble beat.

Single-beat fiducials (dicrotic notch, tidal wave) are fragile on a wrist PPG.
The median of all *accepted* beats (time-normalised to the median IBI) is far
more stable, so morphology is measured on that ensemble beat. Every morphology
feature is dimensionless or in ms, never in ADC counts, so it can be compared
across people.

Second-derivative (APG) a-e waves follow Takazawa et al. (1998) / Elgendi (2012).
"""
import numpy as np
from scipy.signal import find_peaks, resample, savgol_filter

from .quality import detrended_beat


def ensemble_beat(x, feet, accepted, fs, up_fs=1000.0):
    """
    Median accepted beat, resampled to the median IBI at `up_fs`.
    Returns (t_ms, wave 0..1, n_beats) or (None, None, 0).
    """
    idx = np.flatnonzero(accepted)
    if len(idx) < 3:
        return None, None, 0
    ibi = np.median(np.diff(feet)[idx]) / fs
    L = int(round(ibi * up_fs))
    beats = []
    for i in idx:
        b = detrended_beat(x[feet[i]:feet[i + 1] + 1])
        b = resample(b, L)
        rng = b.max() - b.min()
        if rng > 0:
            beats.append((b - b.min()) / rng)
    wave = np.median(beats, axis=0)
    wave = savgol_filter(wave, max(5, int(0.02 * up_fs) | 1), 3)  # 20 ms smoothing at 1 kHz
    wave = (wave - wave.min()) / (wave.max() - wave.min())
    return np.arange(L) / up_fs * 1000.0, wave, len(beats)


def morphology_features(t_ms, wave):
    f = {}
    if wave is None:
        return f
    dt = (t_ms[1] - t_ms[0]) / 1000.0
    ibi_ms = t_ms[-1]
    vpg = np.gradient(wave, dt)
    apg = np.gradient(vpg, dt)

    sp = int(np.argmax(wave[: int(0.6 * len(wave))]))  # systolic peak
    f["rise_time_ms"] = t_ms[sp]
    f["crest_time_ratio"] = t_ms[sp] / ibi_ms
    above = np.flatnonzero(wave >= 0.5)
    f["width_50_ms"] = (above[-1] - above[0]) * dt * 1000 if len(above) else np.nan
    f["max_upslope_per_s"] = float(vpg[:sp + 1].max()) if sp > 0 else np.nan

    # dicrotic notch: first local minimum after the peak, else the APG e-wave (inflection)
    tail = wave[sp: int(0.85 * len(wave))]
    mins, _ = find_peaks(-tail)
    notch = None
    if len(mins):
        notch = sp + int(mins[0])
        f["notch_type"] = "minimum"
    else:
        e_peaks, _ = find_peaks(apg[sp: int(0.85 * len(wave))])
        if len(e_peaks):
            notch = sp + int(e_peaks[np.argmax(apg[sp + e_peaks])])
            f["notch_type"] = "inflection"
    if notch is not None:
        f["notch_time_ms"] = t_ms[notch]
        f["notch_rel_height"] = float(wave[notch])  # wave is 0..1, so already h_notch/h_peak
        maxs, _ = find_peaks(wave[notch:])
        if len(maxs):
            dp = notch + int(maxs[0])
            f["diastolic_rel_height"] = float(wave[dp])   # reflection index
            f["delta_t_ms"] = t_ms[dp] - t_ms[sp]         # Stiffness Index = height / delta_t
        area = np.trapezoid(wave, dx=dt)
        f["systolic_area_ratio"] = float(np.trapezoid(wave[:notch], dx=dt) / area) if area > 0 else np.nan

    # APG a/b/c/d/e waves on the systolic upstroke + early decline
    a_i = int(np.argmax(apg[:sp + 1])) if sp > 0 else 0
    a = apg[a_i]
    ext_min, _ = find_peaks(-apg[a_i:])
    ext_max, _ = find_peaks(apg[a_i:])
    if a > 0 and len(ext_min):
        b_i = a_i + ext_min[0]
        f["apg_b_a"] = apg[b_i] / a
        later_max = ext_max[ext_max + a_i > b_i]
        if len(later_max):
            c_i = a_i + later_max[0]
            f["apg_c_a"] = apg[c_i] / a
            later_min = ext_min[ext_min + a_i > c_i]
            if len(later_min):
                d_i = a_i + later_min[0]
                f["apg_d_a"] = apg[d_i] / a
                e_cand = ext_max[ext_max + a_i > d_i]
                if len(e_cand):
                    f["apg_e_a"] = apg[a_i + e_cand[0]] / a
                    # Takazawa ageing index (b - c - d - e)/a
                    f["apg_aging_index"] = f["apg_b_a"] - f["apg_c_a"] - f["apg_d_a"] - f["apg_e_a"]
    return f


def rhythm_features(table):
    """HRV-style features from consecutive *accepted* beats only."""
    acc = table["accepted"]
    ibi = table["ibi_s"] * 1000.0
    f = {}
    if acc.sum() < 3:
        return f
    f["hr_bpm"] = 60000.0 / np.median(ibi[acc])
    f["sdnn_ms"] = float(np.std(ibi[acc], ddof=1))
    f["cvnn_pct"] = 100.0 * f["sdnn_ms"] / np.mean(ibi[acc])
    pair = acc[1:] & acc[:-1]
    if pair.sum() >= 2:
        f["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(ibi)[pair] ** 2)))
    return f


def perfusion_features(table):
    """Perfusion index (AC/DC, %) of accepted beats: the 'volume' (Akriti) proxy."""
    acc = table["accepted"]
    if acc.sum() < 3:
        return {}
    f = {"perfusion_index_pct": float(np.median(table["amplitude"][acc]))}
    if np.isfinite(table["force_mean"]).any():
        f["contact_force_raw"] = float(np.nanmedian(table["force_mean"][acc]))
    return f
