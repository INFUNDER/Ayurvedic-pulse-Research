"""Run with:  python -m pytest tests   (or: python tests/test_pipeline.py)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse_core import PipelineConfig, load_session, run_pipeline  # noqa: E402
from pulse_core.preprocess import cancel_contact_force  # noqa: E402
from pulse_core.pressure import pressure_sweep_calibration  # noqa: E402
from pulse_core.synth import SubjectParams, clean_pulse, make_session, oscillometric_gain  # noqa: E402

FS = 100.0


def _true_hr(truth):
    return 60.0 / np.median(np.diff(truth["peaks"]) / FS)


def test_hr_tracking_across_subjects():
    for seed in range(6):
        s, truth = make_session(duration_s=60, seed=seed)
        res = run_pipeline(s)
        assert abs(np.median(res.hr_track.hr_bpm) - _true_hr(truth)) < 3, seed


def test_accepted_beats_are_clean_and_hr_is_right():
    s, truth = make_session(duration_s=90, seed=3, snr_db=15, motion_bursts=4)
    res = run_pipeline(s)
    for ch in res.channels:
        assert ch.acceptance > 0.4
        assert abs(ch.features["hr_bpm"] - _true_hr(truth)) < 3
        assert ch.table["corr"][ch.table["accepted"]].min() >= ch.table["corr_thr"]


def test_motion_bursts_get_rejected():
    s, truth = make_session(duration_s=90, seed=5, motion_bursts=0, force_events=0)
    clean_rate = run_pipeline(s).channels[0].acceptance
    s2, _ = make_session(duration_s=90, seed=5, motion_bursts=8, force_events=0)
    noisy_rate = run_pipeline(s2).channels[0].acceptance
    assert noisy_rate < clean_rate


def test_force_canceller_keeps_the_pulse():
    """FSR that carries only the arterial pressure pulse must not cancel the PPG pulse."""
    rng = np.random.default_rng(0)
    p = SubjectParams.random(rng)
    pulse, _, _ = clean_pulse(int(60 * FS), FS, p, rng)
    ppg = pulse - pulse.mean()
    fsr = 1500 + 30 * pulse + rng.normal(0, 2, len(pulse))
    out, _ = cancel_contact_force(ppg, fsr, FS, p.hr_bpm / 60)
    assert np.corrcoef(out, ppg)[0, 1] > 0.98
    assert np.std(out) > 0.9 * np.std(ppg)


def test_force_canceller_removes_force_drift():
    rng = np.random.default_rng(1)
    p = SubjectParams.random(rng)
    pulse, _, _ = clean_pulse(int(60 * FS), FS, p, rng)
    t = np.arange(len(pulse)) / FS
    force = 1500 + 300 * np.sin(2 * np.pi * 0.05 * t)
    ppg = (pulse - pulse.mean()) + 0.004 * (force - 1500)
    out, _ = cancel_contact_force(ppg, force, FS, p.hr_bpm / 60)
    assert np.corrcoef(out, pulse)[0, 1] > np.corrcoef(ppg, pulse)[0, 1] + 0.1


def test_pressure_sweep_finds_optimum():
    rng = np.random.default_rng(2)
    f_opt = 1400.0
    force = np.linspace(400, 3500, 300)
    amp = 2.0 * oscillometric_gain(force, f_opt) * rng.uniform(0.9, 1.1, len(force))
    cal = pressure_sweep_calibration(amp, force)
    assert cal is not None and cal["peak_inside_sweep"]
    assert abs(cal["F_opt"] - f_opt) / f_opt < 0.2
    assert cal["F_light"] < cal["F_opt"] < cal["F_deep"]


def test_v2_csv_roundtrip_with_timer_wrap():
    s, truth = make_session(duration_s=40, seed=7, motion_bursts=1, force_events=1)
    n = s.n_samples
    t_us = (np.arange(n) * 10_000 + (2**32 - 50_000)) % 2**32  # wraps after 5 samples
    cols = [t_us]
    for c in range(3):
        cols += [s.ir[c], s.ir[c] * 0.8]
    cols += list(s.fsr)
    header = "t_us,ir_vata,red_vata,ir_pitta,red_pitta,ir_kapha,red_kapha,fsr_vata,fsr_pitta,fsr_kapha"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.csv")
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="", fmt="%.0f")
        s2 = load_session(path)
    assert s2.fs == FS and abs(s2.n_samples - n) <= 1
    assert s2.fsr is not None and s2.red is not None
    res = run_pipeline(s2)
    assert abs(np.median(res.hr_track.hr_bpm) - _true_hr(truth)) < 3


def test_ablation_configs_run():
    s, _ = make_session(duration_s=40, seed=8)
    for cfg in (PipelineConfig(adaptive_filter=False), PipelineConfig(use_wavelet=False),
                PipelineConfig(use_template_sqi=False), PipelineConfig(use_force_cancellation=False)):
        assert len(run_pipeline(s, cfg).channels) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
