"""
Run the adaptive pipeline on a recorded session and report label-free quality.

    python scripts/analyze_session.py data/sessions/nadi_session_20260904_122504.csv
    python scripts/analyze_session.py data/sessions/*.csv --summary-only
    python scripts/analyze_session.py <file> --sweep        # pressure-sweep calibration

Writes results/<name>_report.png (signals, accepted/rejected beats, ensemble
beats, HR track) and results/<name>_features.csv.

Label-free quality metrics printed per channel:
  accepted %     beats passing the subject-adaptive quality gate
  median corr    beat-vs-own-template correlation
  repeatability  CV of key features across 30 s windows (lower = more stable);
                 a trait like Prakriti needs features that are stable within a
                 person before any classifier is worth training.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse_core import load_session, run_pipeline  # noqa: E402
from pulse_core.pressure import pressure_sweep_calibration  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPEAT_FEATURES = ("hr_bpm", "crest_time_ratio", "notch_rel_height", "systolic_area_ratio", "perfusion_index_pct")


def repeatability(session, win_s=30.0):
    rows = []
    for s in np.arange(0, session.duration_s - win_s + 1, win_s):
        res = run_pipeline(session.slice(s, s + win_s))
        for ch in res.channels:
            rows.append({"channel": ch.name, "start_s": s, **{k: ch.features.get(k, np.nan) for k in REPEAT_FEATURES}})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    cv = lambda v: float(np.nanstd(v) / abs(np.nanmean(v))) if np.isfinite(v).sum() >= 2 else np.nan
    return {ch: {k: cv(g[k].to_numpy(float)) for k in REPEAT_FEATURES} for ch, g in df.groupby("channel")}


def plot_report(session, res, path, span_s=20.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = len(res.channels)
    fig = plt.figure(figsize=(16, 3.2 * (C + 1)))
    gs = fig.add_gridspec(C + 1, 4)
    a0 = int(max(0, session.duration_s / 2 - span_s / 2) * res.fs)
    sl = slice(a0, a0 + int(span_s * res.fs))
    t = np.arange(session.n_samples) / res.fs
    for c, ch in enumerate(res.channels):
        ax = fig.add_subplot(gs[c, :3])
        ax.plot(t[sl], ch.pulsatile[sl] - np.median(ch.pulsatile[sl]), color="0.75", lw=0.7, label="AC/DC input")
        ax.plot(t[sl], ch.clean[sl], lw=0.9, label="adaptive clean")
        tab = ch.table
        for i in range(len(tab.get("accepted", []))):
            a, b = tab["foot"][i], ch.feet[i + 1]
            if a >= sl.start and b <= sl.stop:
                ax.axvspan(t[a], t[b], color="tab:green" if tab["accepted"][i] else "tab:red", alpha=0.10, lw=0)
        ax.set_title(f"{ch.name}: {100 * ch.acceptance:.0f}% beats accepted (green), corr thr {tab.get('corr_thr', np.nan):.2f}")
        ax.set_ylabel("% AC/DC")
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax2 = fig.add_subplot(gs[c, 3])
        if ch.ensemble is not None:
            ax2.plot(ch.ensemble_t_ms, ch.ensemble, lw=1.5)
            ax2.set_title(f"ensemble beat (n={ch.features.get('n_ensemble_beats', 0)})")
            ax2.set_xlabel("ms")
    ax = fig.add_subplot(gs[C, :3])
    ax.plot(res.hr_track.t, res.hr_track.hr_bpm, label="HR track")
    ax.set_ylabel("bpm")
    ax.set_xlabel("time (s)")
    axc = ax.twinx()
    axc.plot(res.hr_track.t, res.hr_track.confidence, color="tab:orange", alpha=0.6, label="confidence")
    axc.set_ylim(0, 1)
    if session.fsr is not None:
        axf = fig.add_subplot(gs[C, 3])
        for c, name in enumerate(session.channels):
            axf.plot(t, session.fsr[c], lw=0.7, label=name)
        axf.set_title("FSR contact force (raw)")
        axf.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="treat the recording as a pressure sweep")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    for path in args.files:
        name = os.path.splitext(os.path.basename(path))[0]
        s = load_session(path)
        res = run_pipeline(s)
        print(f"\n=== {name}  ({s.duration_s:.0f} s @ {s.fs:.0f} Hz, FSR: {'yes' if s.fsr is not None else 'no'})")
        print(f"HR track median {np.median(res.hr_track.hr_bpm):.1f} bpm, median confidence {np.median(res.hr_track.confidence):.2f}")
        print(pd.DataFrame(res.summary()).to_string(index=False))
        if args.summary_only:
            continue

        rep = repeatability(s)
        if rep:
            print("feature CV across 30 s windows:")
            print(pd.DataFrame(rep).T.round(3).to_string())
        feats = pd.DataFrame({ch.name: ch.features for ch in res.channels}).T
        feats.to_csv(os.path.join(out_dir, f"{name}_features.csv"))
        plot_report(s, res, os.path.join(out_dir, f"{name}_report.png"))
        print(f"wrote results/{name}_report.png and results/{name}_features.csv")

        if args.sweep:
            if s.fsr is None:
                print("pressure sweep needs FSR data (v2 firmware)")
                continue
            for ch in res.channels:
                t = ch.table
                cal = pressure_sweep_calibration(t["amplitude"][t["accepted"]], t["force_mean"][t["accepted"]])
                if cal is None:
                    print(f"{ch.name}: not enough accepted beats across forces for a calibration")
                else:
                    print(f"{ch.name}: F_light {cal['F_light']:.0f}  F_opt {cal['F_opt']:.0f}  F_deep {cal['F_deep']:.0f}  "
                          f"width {cal['width_rel']:.2f}  peak inside sweep: {cal['peak_inside_sweep']}")


if __name__ == "__main__":
    main()
