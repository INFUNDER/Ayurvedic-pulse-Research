"""
Synthetic 3-site radial PPG + FSR with known ground truth, for benchmarking
the denoiser before any labelled human data exists.

Each synthetic *subject* draws its own HR, HRV, pulse morphology (reflected
wave timing/size, dicrotic notch depth), perfusion and optimal contact force,
so a method that only works with one person's numbers fails visibly.

Noise model (all optional):
  * respiration-coupled baseline wander + slow drift
  * contact force F(t): strap baseline, drift, press/shift events. Force moves
    the optical DC and scales the pulse amplitude through an oscillometric gain
    curve g(F); the FSR measures F(t) + a small arterial pressure pulse + noise
  * motion bursts shared across sites with site-specific gain
  * white sensor noise at a given SNR
  * I2C dropouts (zeros)
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .io import CHANNELS, from_arrays


@dataclass
class SubjectParams:
    hr_bpm: float
    rsa_bpm: float          # respiratory sinus arrhythmia depth
    resp_hz: float
    refl_delay: float       # reflected wave delay, fraction of IBI
    refl_amp: float         # reflected/diastolic wave amplitude relative to systolic
    sys_width: float        # systolic Gaussian width, fraction of IBI
    perfusion_pct: float    # AC/DC at optimal force
    f_opt: float            # optimal hold-down force (FSR counts)

    @staticmethod
    def random(rng):
        return SubjectParams(
            hr_bpm=rng.uniform(50, 105), rsa_bpm=rng.uniform(1, 6), resp_hz=rng.uniform(0.15, 0.35),
            refl_delay=rng.uniform(0.25, 0.42), refl_amp=rng.uniform(0.25, 0.7),
            sys_width=rng.uniform(0.06, 0.10), perfusion_pct=rng.uniform(0.3, 3.0),
            f_opt=rng.uniform(900, 2200))


def _beat_shape(phase, p, site_shift=0.0):
    """Three-Gaussian beat (systolic, reflected/tidal, diastolic) on phase 0..1."""
    wrap = lambda mu: (phase - mu + 0.5) % 1.0 - 0.5  # periodic distance: no step at beat onset
    g = lambda mu, sd, a: a * np.exp(-0.5 * (wrap(mu) / sd) ** 2)
    sys_mu = 0.16 + site_shift
    return (g(sys_mu, p.sys_width, 1.0)
            + g(sys_mu + 0.12, p.sys_width * 1.4, 0.35 * p.refl_amp)
            + g(sys_mu + p.refl_delay, p.sys_width * 1.6, p.refl_amp))


def clean_pulse(n, fs, p, rng, site_shift=0.0):
    """Returns (pulse 0..1-ish, beat onset indices, systolic peak indices)."""
    t = np.arange(n) / fs
    inst_hr = p.hr_bpm + p.rsa_bpm * np.sin(2 * np.pi * p.resp_hz * t) + rng.normal(0, 0.5, n).cumsum() / np.sqrt(fs * 30)
    phase_total = np.cumsum(inst_hr / 60.0) / fs
    phase = phase_total % 1.0
    pulse = _beat_shape(phase, p, site_shift)
    onsets = np.flatnonzero(np.diff(np.floor(phase_total)) > 0) + 1
    peaks = np.array([o + int(np.argmax(pulse[o:o + int(0.5 * fs)])) for o in onsets if o + int(0.5 * fs) < n])
    return pulse, onsets, peaks


def oscillometric_gain(force, f_opt, width=0.6):
    """Pulse amplitude vs hold-down force: peaks at f_opt, log-Gaussian shape."""
    return np.exp(-0.5 * (np.log(np.maximum(force, 1) / f_opt) / width) ** 2)


def make_session(duration_s=120, fs=100.0, seed=0, snr_db=20.0, wander=1.0, force_events=4,
                 motion_bursts=3, dropouts=True, params=None, base_pulse=None):
    """
    Returns (Session, truth) where truth has the clean pulsatile signal per channel,
    systolic peak indices (from channel 0) and the subject parameters.

    base_pulse: optional real clean PPG at `fs` (e.g. VitalDB) used instead of the
    Gaussian beat model; its peaks become the ground truth.
    """
    rng = np.random.default_rng(seed)
    p = params or SubjectParams.random(rng)
    t = np.arange(int(duration_s * fs)) / fs

    if base_pulse is None:
        pulse0, onsets, peaks = clean_pulse(len(t), fs, p, rng)
    else:
        b = np.asarray(base_pulse, float)[: len(t)]
        t = t[: len(b)]
        pulse0 = (b - b.min()) / np.ptp(b)
        peaks, _ = find_peaks(pulse0, distance=int(0.33 * fs), prominence=0.3 * np.std(pulse0))
        onsets = np.array([peaks[i - 1] + int(np.argmin(pulse0[peaks[i - 1]:peaks[i]])) for i in range(1, len(peaks))])
    n = len(t)
    # common strap tension + per-site offset; press / shift events with exponential recovery
    strap = p.f_opt * rng.uniform(0.8, 1.2) + 80 * np.sin(2 * np.pi * rng.uniform(0.005, 0.02) * t)
    events = np.zeros(n)
    for _ in range(force_events):
        s = rng.integers(int(5 * fs), n - int(5 * fs))
        events[s:] += rng.uniform(-0.35, 0.35) * p.f_opt * np.exp(-(t[s:] - t[s]) / rng.uniform(1.5, 6.0))

    ir, fsr, clean = [], [], []
    for c in range(3):
        site_force = strap * rng.uniform(0.75, 1.25) + events * rng.uniform(0.6, 1.0)
        gain = oscillometric_gain(site_force, p.f_opt * rng.uniform(0.85, 1.15))
        pulse = np.roll(pulse0, c)  # ~1 sample transit difference between adjacent sites
        clean_ac = p.perfusion_pct * gain * (pulse - pulse.mean())
        clean.append(clean_ac)
        dc = 60000 * (1 + 0.25 * (site_force / p.f_opt - 1))           # pressing changes optical DC
        drift = wander * p.perfusion_pct * (0.8 * np.sin(2 * np.pi * p.resp_hz * t)
                                            + 1.5 * np.sin(2 * np.pi * 0.03 * t + c))
        ppg_pct = clean_ac + drift
        sig_pow = np.var(clean_ac)
        ppg_pct = ppg_pct + rng.normal(0, np.sqrt(sig_pow / 10 ** (snr_db / 10)), n)
        ir.append(dc * (1 - ppg_pct / 100.0))  # IR counts fall in systole
        fsr.append(site_force + 0.02 * p.f_opt * pulse + rng.normal(0, 4, n))
    ir, fsr = np.array(ir), np.array(fsr)

    for _ in range(motion_bursts):  # gross motion: shared shape, site-specific gain
        s = rng.integers(0, n - int(3 * fs))
        L = int(rng.uniform(1.0, 3.0) * fs)
        burst = np.hanning(L) * np.sin(2 * np.pi * rng.uniform(0.8, 3.0) * np.arange(L) / fs)
        for c in range(3):
            g = rng.uniform(0.5, 1.5) * 3 * p.perfusion_pct / 100
            ir[c, s:s + L] *= 1 + g * burst
            fsr[c, s:s + L] += 0.2 * p.f_opt * g / (3 * p.perfusion_pct / 100) * burst

    if dropouts:
        for c in range(3):
            idx = rng.choice(n, size=int(0.002 * n), replace=False)
            ir[c, idx] = 0

    session = from_arrays(ir, fs, fsr=fsr, channels=CHANNELS, source=f"synthetic seed={seed}")
    truth = {"clean": np.array(clean), "peaks": peaks, "onsets": onsets, "params": p}
    return session, truth
