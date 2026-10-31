"""
Stage 4 - HR-informed beat detection.

Instead of one global prominence threshold, both the minimum peak spacing and
the prominence requirement are local:
  * prominence >= 35 % of the local pulse amplitude envelope
    (90th-10th percentile over 3 s), so amplitude changes from pressure or
    vasomotion don't make beats vanish;
  * two peaks closer than 60 % of the *expected* inter-beat interval (from the
    HR track) cannot both be systolic peaks; the larger one is kept.
"""
import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks


def detect_beats(x, fs, hr_track, prominence_frac=0.35, envelope_s=3.0):
    """Returns (peaks, feet) as integer index arrays of equal length."""
    n = len(x)
    ibi = 60.0 / hr_track.per_sample(n, fs) * fs  # expected IBI in samples
    size = max(3, int(envelope_s * fs))
    amp = percentile_filter(x, 90, size=size) - percentile_filter(x, 10, size=size)
    min_dist = max(1, int(0.33 * ibi.min()))
    cand, _ = find_peaks(x, distance=min_dist, prominence=prominence_frac * amp)

    peaks = []
    for p in cand:
        if peaks and p - peaks[-1] < 0.6 * ibi[p]:
            if x[p] > x[peaks[-1]]:
                peaks[-1] = p
            continue
        peaks.append(p)
    peaks = np.asarray(peaks, int)

    feet = np.empty(len(peaks), int)
    for i, p in enumerate(peaks):
        lo = max(0, int(p - 0.6 * ibi[p]), peaks[i - 1] + 1 if i else 0)
        feet[i] = lo + int(np.argmin(x[lo:p + 1])) if p > lo else p
    return peaks, feet
