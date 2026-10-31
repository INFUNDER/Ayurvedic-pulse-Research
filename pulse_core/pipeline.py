"""
Adaptive Nadi pulse pipeline: raw 3x PPG + 3x FSR  ->  clean beats + features.

    Session (raw counts)
      1. to_pulsatile            invert + AC/DC  -> person-independent units (%)
      2. track_heart_rate        3-channel spectral fusion + Viterbi -> f0(t)
      3. cancel_contact_force    FSR (low-passed below f0) as adaptive reference
      4. adaptive_denoise        HP(0.5 f0) -> BayesShrink wavelet -> LP(8 f0)
      5. detect_beats            HR-informed peaks / feet
      6. assess_beats            per-subject template + robust z-scores + cross-channel
      7. features                ensemble beat morphology, rhythm, perfusion

Every stage can be switched off in `PipelineConfig` for ablation studies.
Channels are processed independently after HR tracking. The three sites are
NOT averaged or common-mode subtracted, because the pulse itself is the
dominant common-mode component (adjacent sites on one artery correlate > 0.9).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .adaptive_filter import adaptive_denoise, fixed_bandpass
from .beats import detect_beats
from .features import ensemble_beat, morphology_features, perfusion_features, rhythm_features
from .hr_tracker import HRTrack, track_heart_rate
from .io import Session
from .preprocess import cancel_contact_force, to_pulsatile
from .quality import SubjectTemplate, assess_beats


@dataclass
class PipelineConfig:
    hr_range: tuple = (40, 180)
    use_force_cancellation: bool = True
    adaptive_filter: bool = True       # False -> fixed 0.5-10 Hz band-pass (v1 behaviour)
    use_wavelet: bool = True
    hp_factor: float = 0.5
    n_harmonics: int = 8
    use_template_sqi: bool = True      # False -> accept every detected beat
    use_cross_channel: bool = True
    corr_floor: float = 0.85
    corr_ceiling: float = 0.92
    robust_k: float = 3.5
    template_alpha: float = 0.05


@dataclass
class ChannelResult:
    name: str
    pulsatile: np.ndarray
    clean: np.ndarray
    peaks: np.ndarray
    feet: np.ndarray
    table: dict
    template: SubjectTemplate
    ensemble_t_ms: Optional[np.ndarray]
    ensemble: Optional[np.ndarray]
    features: dict
    noise_sigma: float = 0.0
    force_artifact: Optional[np.ndarray] = None

    @property
    def acceptance(self):
        acc = self.table.get("accepted", np.zeros(0, bool))
        return float(acc.mean()) if len(acc) else 0.0


@dataclass
class PipelineResult:
    fs: float
    hr_track: HRTrack
    channels: list
    config: PipelineConfig

    def summary(self):
        rows = []
        for ch in self.channels:
            t = ch.table
            acc = t.get("accepted", np.zeros(0, bool))
            reasons = {}
            for r in t.get("reason", []):
                if r:
                    reasons[r] = reasons.get(r, 0) + 1
            rows.append({
                "channel": ch.name,
                "beats": int(len(acc)),
                "accepted_pct": round(100 * ch.acceptance, 1),
                "hr_bpm": round(ch.features.get("hr_bpm", np.nan), 1),
                "median_corr": round(float(np.median(t["corr"])) if len(acc) else np.nan, 3),
                "corr_thr": round(t.get("corr_thr", np.nan), 3),
                "perfusion_pct": round(ch.features.get("perfusion_index_pct", np.nan), 3),
                "rejections": reasons,
            })
        return rows


@dataclass
class PipelineState:
    """Carry per-channel templates across windows in live mode."""
    templates: dict = field(default_factory=dict)


def run_pipeline(session: Session, config: PipelineConfig = None,
                 state: PipelineState = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    fs = session.fs
    C, N = session.ir.shape

    pulsatile, _ = to_pulsatile(session.ir, fs)
    hr = track_heart_rate(pulsatile, fs, session.valid, cfg.hr_range)

    stage = []
    for c in range(C):
        x = pulsatile[c]
        artifact = None
        if cfg.use_force_cancellation and session.fsr is not None:
            x, artifact = cancel_contact_force(x, session.fsr[c], fs, hr.hr_bpm / 60.0)
        if cfg.adaptive_filter:
            y, sigma = adaptive_denoise(x, fs, hr, cfg.hp_factor, cfg.n_harmonics, cfg.use_wavelet)
        else:
            y, sigma = fixed_bandpass(x, fs), 0.0
        peaks, feet = detect_beats(y, fs, hr)
        stage.append((x, y, peaks, feet, sigma, artifact))

    # a channel takes part in cross-channel checks only if it found >= half the expected beats
    expected_beats = N / fs * hr.median_hz
    usable = [len(s[2]) >= 0.5 * expected_beats for s in stage]

    results = []
    for c in range(C):
        x, y, peaks, feet, sigma, artifact = stage[c]
        name = session.channels[c]
        tmpl = (state.templates.get(name) if state else None) or SubjectTemplate(alpha=cfg.template_alpha)
        others = [stage[o][2] for o in range(C) if o != c and usable[o]] if cfg.use_cross_channel else []
        fsr = session.fsr[c] if session.fsr is not None else None
        table, _ = assess_beats(y, fs, peaks, feet, hr, session.valid[c], tmpl, fsr, others,
                                cfg.corr_floor, cfg.corr_ceiling, cfg.robust_k)
        if not cfg.use_template_sqi and len(table["accepted"]):
            table["accepted"] = table["valid_frac"] >= 0.95
        if state is not None:
            state.templates[name] = tmpl

        feats = {}
        t_ms, ens = None, None
        if len(table["accepted"]):
            t_ms, ens, n_ens = ensemble_beat(y, feet, table["accepted"], fs)
            feats.update(rhythm_features(table))
            feats.update(perfusion_features(table))
            feats.update(morphology_features(t_ms, ens))
            feats["n_ensemble_beats"] = n_ens
        results.append(ChannelResult(name, pulsatile[c], y, peaks, feet, table, tmpl, t_ms, ens,
                                     feats, sigma, artifact))
    return PipelineResult(fs, hr, results, cfg)
