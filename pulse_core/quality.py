"""
Stage 5 - subject-adaptive beat quality assessment.

The rule that replaces fixed thresholds: **only threshold quantities that are
already normalised to the person.**

  * Shape   - correlation of each beat with the subject's *own* running template
              (adaptive template matching; cf. Li & Clifford 2012, Lim et al. 2018).
              The template is bootstrapped from the most mutually similar beats of
              the recording, then updated slowly with accepted beats only.
  * Size    - robust z-score of log pulse amplitude against this recording's own
              median/MAD, not an absolute amplitude.
  * Timing  - IBI relative to the tracked HR. Only physiologically implausible
              ratios (missed/extra beat) are rejected, so genuine rhythm
              irregularity (Tala, a Vata marker) survives.
  * Contact - robust z-score of FSR force change during the beat.
  * Space   - the same beat must appear on at least one other sensor.

The correlation cut-off itself adapts: median(corr) - k*MAD, clipped to
[corr_floor, corr_ceiling] = [0.85, 0.92]; a beat matching the subject's own
template at r >= 0.92 is never rejected for shape.
"""
import numpy as np
from scipy.signal import resample

N_TEMPLATE = 64


def robust_z(v):
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    return (v - med) / (mad if mad > 1e-12 else (np.nanstd(v) + 1e-12))


def detrended_beat(seg):
    """Remove the foot-to-next-foot line so the beat starts and ends at 0."""
    ramp = np.linspace(seg[0], seg[-1], len(seg))
    return seg - ramp


def normalise_beat(seg, n=N_TEMPLATE):
    """Time-normalise to n samples and scale to 0..1 (shape only)."""
    b = resample(detrended_beat(seg), n)
    rng = b.max() - b.min()
    return (b - b.min()) / rng if rng > 1e-12 else np.zeros(n)


def _corr_rows(a, t):
    a = a - a.mean(axis=1, keepdims=True)
    t = t - t.mean()
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(t)
    return (a @ t) / np.where(den > 0, den, np.inf)


class SubjectTemplate:
    """Per-subject, per-channel beat template that adapts as the session goes on."""

    def __init__(self, n=N_TEMPLATE, alpha=0.05):
        self.n = n
        self.alpha = alpha
        self.wave = None

    def bootstrap(self, beats, max_beats=300):
        """Median of the most mutually similar beats (robust to a noisy start)."""
        b = beats[:max_beats]
        if len(b) < 5:
            return False
        c = np.corrcoef(b)
        score = np.nanmedian(c, axis=1)
        core = b[score >= np.nanpercentile(score, 50)]
        tmpl = np.median(core, axis=0)
        for _ in range(2):  # refine against the provisional template
            r = _corr_rows(b, tmpl)
            sel = b[r >= np.nanpercentile(r, 50)]
            tmpl = np.median(sel, axis=0)
        self.wave = tmpl
        return True

    def correlate(self, beats):
        return _corr_rows(beats, self.wave)

    def update(self, beats, accepted):
        for b in beats[accepted]:
            self.wave = (1 - self.alpha) * self.wave + self.alpha * b


def cross_channel_agreement(peaks, other_peaks, fs, tol_s=0.08):
    """For each peak, does any other usable channel have a peak within +-tol?"""
    if not other_peaks:
        return np.ones(len(peaks), bool)
    tol = tol_s * fs
    ok = np.zeros(len(peaks), bool)
    for op in other_peaks:
        if len(op) == 0:
            continue
        idx = np.clip(np.searchsorted(op, peaks), 1, len(op) - 1) if len(op) > 1 else np.zeros(len(peaks), int)
        d = np.minimum(np.abs(op[idx] - peaks), np.abs(op[np.maximum(idx - 1, 0)] - peaks))
        ok |= d <= tol
    return ok


def assess_beats(x, fs, peaks, feet, hr_track, valid, template, fsr=None, other_peaks=None,
                 corr_floor=0.85, corr_ceiling=0.92, k=3.5, ibi_ratio=(0.6, 1.6)):
    """
    Score every foot-to-foot beat. Returns a dict of per-beat arrays plus the
    (B, N_TEMPLATE) normalised beats. Beat i spans feet[i]..feet[i+1].
    """
    nb = max(len(feet) - 1, 0)
    if nb == 0:
        return {"accepted": np.zeros(0, bool)}, np.zeros((0, N_TEMPLATE))

    segs = [x[feet[i]:feet[i + 1] + 1] for i in range(nb)]
    beats = np.array([normalise_beat(s) if len(s) > 4 else np.zeros(N_TEMPLATE) for s in segs])
    amp = np.array([x[peaks[i]] - min(x[feet[i]], x[feet[i + 1]]) for i in range(nb)])
    ibi = np.diff(feet) / fs
    expected = 60.0 / hr_track.at(feet[:-1] / fs)
    vfrac = np.array([valid[feet[i]:feet[i + 1] + 1].mean() for i in range(nb)])

    reasons = np.full(nb, "", dtype=object)
    reject = np.zeros(nb, bool)

    def flag(mask, why):
        nonlocal reject
        reasons[mask & ~reject] = why
        reject |= mask

    flag(vfrac < 0.95, "dropout")
    flag((ibi / expected < ibi_ratio[0]) | (ibi / expected > ibi_ratio[1]), "ibi")

    if template.wave is None:
        template.bootstrap(beats[~reject])
    corr = template.correlate(beats) if template.wave is not None else np.zeros(nb)
    good = ~reject
    if good.sum() >= 5:
        thr = np.median(corr[good]) - k * 1.4826 * np.median(np.abs(corr[good] - np.median(corr[good])))
    else:
        thr = corr_floor
    corr_thr = float(np.clip(thr, corr_floor, corr_ceiling))
    flag(corr < corr_thr, "shape")

    amp_z = robust_z(np.log(np.maximum(amp, 1e-9)))
    flag(np.abs(amp_z) > k, "amplitude")

    force_mean = np.full(nb, np.nan)
    if fsr is not None:
        force_mean = np.array([fsr[feet[i]:feet[i + 1] + 1].mean() for i in range(nb)])
        dforce = np.array([np.ptp(fsr[feet[i]:feet[i + 1] + 1]) for i in range(nb)])
        # relative floor so ADC quantisation on a perfectly steady strap is not "motion"
        flag((robust_z(dforce) > k) & (dforce > 0.02 * np.nanmedian(np.abs(force_mean))), "contact")

    agree = cross_channel_agreement(peaks[:nb], other_peaks or [], fs)
    flag(~agree, "no_cross_channel")

    accepted = ~reject
    if template.wave is not None:
        template.update(beats, accepted)

    table = dict(foot=feet[:nb], peak=peaks[:nb], ibi_s=ibi, expected_ibi_s=expected,
                 amplitude=amp, corr=corr, amp_z=amp_z, valid_frac=vfrac,
                 force_mean=force_mean, accepted=accepted, reason=reasons, corr_thr=corr_thr)
    return table, beats
