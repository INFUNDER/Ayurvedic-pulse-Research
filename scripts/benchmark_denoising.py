"""
Label-free validation of the adaptive denoiser.

No dosha labels are needed to prove that noise cancellation works: inject
realistic artifacts into signals whose clean version is known, then measure how
well each method recovers beats and morphology.

Datasets
  synthetic : N random virtual subjects (pulse_core.synth), moderate + heavy noise
  vitaldb   : real clean clinical PPG (data/vitaldb_clinical) + the same noise model

Methods (ablation ladder)
  v1_fixed        0.5-10 Hz band-pass + global-prominence peaks, all beats used
  adaptive        HR-tracking filter + wavelet, all beats used
  adaptive+sqi    + subject template / robust beat quality
  full            + FSR contact-force cancellation

Usage:  python scripts/benchmark_denoising.py [--subjects 20] [--no-vitaldb]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse_core import PipelineConfig, run_pipeline  # noqa: E402
from pulse_core.adaptive_filter import fixed_bandpass  # noqa: E402
from pulse_core.features import ensemble_beat, morphology_features  # noqa: E402
from pulse_core.quality import normalise_beat  # noqa: E402
from pulse_core.synth import make_session  # noqa: E402

FS = 100.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = {
    "v1_fixed": None,
    "adaptive": PipelineConfig(use_force_cancellation=False, use_template_sqi=False, use_cross_channel=False),
    "adaptive+sqi": PipelineConfig(use_force_cancellation=False),
    "full": PipelineConfig(),
}
NOISE = {
    "moderate": dict(snr_db=20, wander=1.0, force_events=4, motion_bursts=3),
    "heavy": dict(snr_db=10, wander=2.5, force_events=8, motion_bursts=6),
}


def v1_baseline(session):
    """Re-implementation of the v1 approach (fixed band-pass, global threshold)."""
    x = -fixed_bandpass(session.ir[0], session.fs)
    peaks, _ = find_peaks(x, distance=int(0.3 * session.fs), prominence=0.5 * np.std(x))
    w = int(0.3 * session.fs)
    feet = np.array([max(0, p - w) + int(np.argmin(x[max(0, p - w):p + 1])) for p in peaks], int)
    return x, peaks, feet, np.ones(max(len(feet) - 1, 0), bool)


def match(det, ref, tol):
    """Greedy one-to-one matching of peak indices; returns matched mask over det."""
    ok = np.zeros(len(det), bool)
    used = np.zeros(len(ref), bool)
    for i, d in enumerate(det):
        j = np.searchsorted(ref, d)
        for k in (j - 1, j):
            if 0 <= k < len(ref) and not used[k] and abs(ref[k] - d) <= tol:
                ok[i] = used[k] = True
                break
    return ok


def evaluate(x, peaks, feet, accepted, truth, fs):
    ref_peaks = truth["peaks"]
    clean = truth["clean"][0]
    tol = int(0.06 * fs)
    hit = match(peaks, ref_peaks, tol)
    tp = hit.sum()
    sens = tp / max(len(ref_peaks), 1)
    ppv = tp / max(len(peaks), 1)

    nb = len(accepted)
    beat_ok = np.zeros(nb, bool)
    for i in range(nb):
        a, b = feet[i], feet[i + 1]
        if b - a > 5:
            r = np.corrcoef(normalise_beat(x[a:b + 1]), normalise_beat(clean[a:b + 1]))[0, 1]
            beat_ok[i] = hit[i] and r > 0.9
    acc = accepted.astype(bool)

    # HR in 10 s windows from accepted beats vs truth
    errs = []
    for s in np.arange(0, len(clean) / fs - 10, 10):
        m = acc & (feet[:-1] / fs >= s) & (feet[:-1] / fs < s + 10)
        tr = ref_peaks[(ref_peaks / fs >= s) & (ref_peaks / fs < s + 10)]
        if m.sum() >= 3 and len(tr) >= 3:
            errs.append(abs(60 / np.median(np.diff(feet)[m] / fs) - 60 / np.median(np.diff(tr) / fs)))

    tf = truth["features"]
    t_ms, ens, _ = ensemble_beat(x, feet, acc, fs)
    f = morphology_features(t_ms, ens) if ens is not None else {}
    ens_corr = np.nan
    if ens is not None:
        ens_corr = np.corrcoef(resample_poly(ens, 200, len(ens)) if len(ens) != 200 else ens,
                               resample_poly(truth["ensemble"], 200, len(truth["ensemble"])))[0, 1]
    return {
        "sensitivity": sens, "ppv": ppv, "f1": 2 * sens * ppv / max(sens + ppv, 1e-9),
        "accepted_clean_precision": beat_ok[acc].mean() if acc.any() else np.nan,
        "yield": acc.sum() / max(len(ref_peaks), 1),
        "hr_mae_bpm": np.mean(errs) if errs else np.nan,
        "ensemble_corr": ens_corr,
        "notch_height_err": abs(f.get("notch_rel_height", np.nan) - tf.get("notch_rel_height", np.nan)),
        "rise_time_err_ms": abs(f.get("rise_time_ms", np.nan) - tf.get("rise_time_ms", np.nan)),
    }


def truth_reference(truth, fs):
    """Ground-truth ensemble beat, using the same foot definition as the pipeline."""
    clean, pk = truth["clean"][0], truth["peaks"]
    feet = np.array([pk[i - 1] + int(np.argmin(clean[pk[i - 1]:pk[i] + 1])) for i in range(1, len(pk))])
    t_ms, ens, _ = ensemble_beat(clean, feet, np.ones(len(feet) - 1, bool), fs)
    truth["ensemble"] = ens
    truth["features"] = morphology_features(t_ms, ens)
    return truth


def run_case(session, truth):
    truth = truth_reference(truth, session.fs)
    rows = {}
    for name, cfg in METHODS.items():
        if cfg is None:
            x, peaks, feet, acc = v1_baseline(session)
        else:
            ch = run_pipeline(session, cfg).channels[0]
            x, peaks, feet, acc = ch.clean, ch.peaks, ch.feet, ch.table["accepted"]
        rows[name] = evaluate(x, peaks, feet, acc, truth, session.fs)
    return rows


def vitaldb_pulses(seconds=120):
    """
    From each local VitalDB case, the 120 s window (first 30 min, resampled to
    100 Hz) with the most regular beats; windows with gaps, flat lines or
    implausible IBIs are skipped, so the ground-truth peaks can be trusted.
    """
    out = []
    L = int(seconds * FS)
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "vitaldb_clinical", "*.npy"))):
        x = resample_poly(np.load(path).astype(float)[: 250 * 1800], 2, 5)  # 250 -> 100 Hz
        best, best_cv = None, 0.15
        for s in range(0, len(x) - L, L // 4):
            seg = x[s:s + L]
            if np.ptp(seg) < 1e-6:
                continue
            z = (seg - seg.min()) / np.ptp(seg)
            pk, _ = find_peaks(z, distance=int(0.33 * FS), prominence=0.3 * np.std(z))
            ibi = np.diff(pk) / FS
            if len(ibi) < 0.5 * seconds or ibi.max() > 1.5 or ibi.max() / ibi.min() > 1.6:
                continue
            cv = ibi.std() / ibi.mean()
            if cv < best_cv:
                best, best_cv = seg, cv
        if best is not None:
            out.append((os.path.basename(path), best))
    return out


def summarise(df):
    metrics = [c for c in df.columns if c not in ("dataset", "noise", "case", "method")]
    g = df.groupby(["dataset", "noise", "method"], sort=False)[metrics]
    higher_better = ("sensitivity", "ppv", "f1", "accepted_clean_precision", "yield", "ensemble_corr")
    mean = g.mean()
    worst = g.agg(lambda v: np.nanpercentile(v, 10 if v.name in higher_better else 90))
    return mean.round(3), worst.round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--no-vitaldb", action="store_true")
    args = ap.parse_args()

    rows = []
    for noise, kw in NOISE.items():
        for seed in range(args.subjects):
            s, truth = make_session(seed=seed, **kw)
            for m, r in run_case(s, truth).items():
                rows.append(dict(dataset="synthetic", noise=noise, case=seed, method=m, **r))
        if not args.no_vitaldb:
            for k, (name, pulse) in enumerate(vitaldb_pulses()):
                s, truth = make_session(seed=100 + k, base_pulse=pulse, **kw)
                for m, r in run_case(s, truth).items():
                    rows.append(dict(dataset="vitaldb", noise=noise, case=name, method=m, **r))
        print(f"finished {noise} noise")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "benchmark_denoising.csv"), index=False)
    mean, worst = summarise(df)
    pd.set_option("display.width", 200)
    print("\n=== mean across subjects ===")
    print(mean.to_string())
    print("\n=== worst-decile subject (10th pct for higher-is-better, 90th for errors) ===")
    print(worst.to_string())
    mean.to_csv(os.path.join(ROOT, "results", "benchmark_summary_mean.csv"))
    worst.to_csv(os.path.join(ROOT, "results", "benchmark_summary_worst.csv"))


if __name__ == "__main__":
    main()
