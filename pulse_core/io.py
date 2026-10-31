"""
Session loading and sample-grid repair.

Supported CSV formats
---------------------
v2 (firmware/nadi_tri_sensor):  t_us, ir_vata, red_vata, ir_pitta, red_pitta,
                                 ir_kapha, red_kapha, fsr_vata, fsr_pitta, fsr_kapha
v1 (old realtime_pipeline.py):   Timestamp, Vata_IR, Pitta_IR, Kapha_IR   (host clock, IR only)
single-sensor:                   Timestamp_ms, Red, IR

Every loader returns a `Session` on a uniform time grid with a per-sample
`valid` mask. Dropped I2C reads (value 0), ADC saturation and timing gaps are
interpolated over but marked invalid, so later stages can refuse to trust them.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

CHANNELS = ("vata", "pitta", "kapha")
MAX30102_FULL_SCALE = 2**18 - 1  # 18-bit ADC


@dataclass
class Session:
    fs: float
    ir: np.ndarray                    # (C, N) raw optical counts
    valid: np.ndarray                 # (C, N) bool
    channels: tuple = CHANNELS
    red: Optional[np.ndarray] = None  # (C, N)
    fsr: Optional[np.ndarray] = None  # (C, N) raw ADC counts, larger = more force
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self):
        return self.ir.shape[1]

    @property
    def duration_s(self):
        return self.n_samples / self.fs

    def slice(self, start_s, stop_s):
        a, b = int(start_s * self.fs), int(stop_s * self.fs)
        cut = lambda x: None if x is None else x[:, a:b]
        return Session(self.fs, self.ir[:, a:b], self.valid[:, a:b], self.channels,
                       cut(self.red), cut(self.fsr), self.source, dict(self.meta))


def _bad_samples(x):
    return (x <= 0) | (x >= MAX30102_FULL_SCALE) | ~np.isfinite(x)


def _fill_invalid(x, bad):
    """Linear interpolation across bad samples (per row)."""
    x = x.astype(float).copy()
    idx = np.arange(x.shape[1])
    for c in range(x.shape[0]):
        good = ~bad[c]
        if good.sum() >= 2:
            x[c, bad[c]] = np.interp(idx[bad[c]], idx[good], x[c, good])
        else:
            x[c, :] = 0.0
    return x


def _resample_to_grid(t, cols, fs, max_gap_s):
    """Resample irregular samples onto a uniform grid; flag grid points inside gaps."""
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.vstack([np.interp(grid, t, c) for c in cols])
    # distance from each grid point to the nearest real sample
    pos = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    gap = np.minimum(np.abs(grid - t[pos - 1]), np.abs(t[pos] - grid))
    return out, gap <= max_gap_s


def from_arrays(ir, fs, fsr=None, red=None, channels=CHANNELS, source="array"):
    ir = np.atleast_2d(np.asarray(ir, float))
    bad = _bad_samples(ir)
    return Session(
        fs=float(fs),
        ir=_fill_invalid(ir, bad),
        valid=~bad,
        channels=tuple(channels[: ir.shape[0]]),
        red=None if red is None else _fill_invalid(np.atleast_2d(red), _bad_samples(np.atleast_2d(red))),
        fsr=None if fsr is None else np.atleast_2d(np.asarray(fsr, float)),
        source=source,
    )


def load_session(path, fs=None, max_gap_s=0.1):
    """Load any supported session CSV into a uniformly sampled `Session`."""
    df = pd.read_csv(path)
    cols = set(df.columns)

    if "t_us" in cols:  # v2: device timestamps, resample on them
        t_us = df["t_us"].to_numpy(float)
        t_us += 2.0**32 * np.concatenate([[0], np.cumsum(np.diff(t_us) < -2.0**31)])  # micros() wraps at ~71 min
        t = t_us * 1e-6
        t, keep = np.unique(t, return_index=True)
        df = df.iloc[keep]
        fs = fs or round(1.0 / np.median(np.diff(t)))
        ir_raw = np.vstack([df[f"ir_{c}"] for c in CHANNELS]).astype(float)
        red_raw = np.vstack([df[f"red_{c}"] for c in CHANNELS]).astype(float)
        fsr_raw = np.vstack([df[f"fsr_{c}"] for c in CHANNELS]).astype(float)
        bad = _bad_samples(ir_raw)
        stacked = np.vstack([_fill_invalid(ir_raw, bad), _fill_invalid(red_raw, _bad_samples(red_raw)),
                             fsr_raw, bad.astype(float)])
        grid, in_range = _resample_to_grid(t, stacked, fs, max_gap_s)
        ir, red, fsr, badg = grid[0:3], grid[3:6], grid[6:9], grid[9:12] > 0
        return Session(fs, ir, ~badg & in_range, CHANNELS, red, fsr, str(path))

    if "Vata_IR" in cols:  # v1: host timestamps are USB-batched, so trust sample order instead
        t = df["Timestamp"].to_numpy(float)
        fs = fs or round(len(t) / (t[-1] - t[0]))
        ir_raw = np.vstack([df[f"{c.capitalize()}_IR"] for c in CHANNELS]).astype(float)
        s = from_arrays(ir_raw, fs, source=str(path))
        s.meta["timing"] = "assumed uniform (v1 host timestamps)"
        return s

    if {"Timestamp_ms", "IR"} <= cols:  # single sensor logger
        t = df["Timestamp_ms"].to_numpy(float) * 1e-3
        t, keep = np.unique(t, return_index=True)
        df = df.iloc[keep]
        fs = fs or round(1.0 / np.median(np.diff(t)))
        ir_raw = df["IR"].to_numpy(float)[None, :]
        bad = _bad_samples(ir_raw)
        grid, in_range = _resample_to_grid(t, np.vstack([_fill_invalid(ir_raw, bad), bad]), fs, max_gap_s)
        return Session(fs, grid[:1], (grid[1:] == 0) & in_range, ("single",), source=str(path))

    raise ValueError(f"Unrecognised session format: {sorted(cols)}")
