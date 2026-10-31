# Ayurvedic Nadi Pariksha – adaptive pulse processing

Wristband with **3× MAX30102** (optical PPG) and **3× FSR402** (contact force)
over the Vata / Pitta / Kapha positions of the radial artery, plus a signal
pipeline that removes noise **without person-specific hand-tuned thresholds**.

> Current phase (see [RESEARCH.md](RESEARCH.md)): build and validate the
> adaptive denoiser *before* any dosha classifier. No dosha labels are needed
> for this phase; the denoiser is validated against known ground truth.

## Layout

```text
pulse_acquisition_system/
├── firmware/nadi_tri_sensor/   ESP32 + TCA9548A: 3x PPG (IR+Red) + 3x FSR @ 100 Hz, start-up auto-gain
├── pulse_core/                 the adaptive pipeline
│   ├── io.py                   load any session CSV -> uniform grid + validity mask
│   ├── preprocess.py           AC/DC normalisation, FSR contact-force cancellation
│   ├── hr_tracker.py           3-channel spectral fusion + Viterbi heart-rate track
│   ├── adaptive_filter.py      HR-tracking band-pass + BayesShrink wavelet denoising
│   ├── beats.py                HR-informed beat detection
│   ├── quality.py              per-subject beat template + robust beat quality
│   ├── features.py             ensemble-beat morphology, rhythm, perfusion
│   ├── pressure.py             pressure-sweep calibration (per-person optimal force)
│   ├── pipeline.py             run_pipeline(): stages wired together, ablation switches
│   └── synth.py                synthetic subjects + realistic noise (ground truth)
├── scripts/
│   ├── record_session.py       serial -> data/sessions/*.csv, optional live quality monitor
│   ├── analyze_session.py      report + figure + feature CSV for any recording
│   ├── benchmark_denoising.py  v1 vs adaptive, synthetic + VitalDB, label-free metrics
│   └── fetch_vitaldb.py        download clean clinical PPG for the benchmark
├── tests/test_pipeline.py
└── data/                       custom_radial/ (tracked), sessions/ + vitaldb_clinical/ (local only)
```

Legacy v1 code (rule-based dosha mapper, Pulse-Mamba model, old scripts) is in
`../_archive/pulse_acquisition_system_v1/` and in git history.

## Quick start

```bash
cd pulse_acquisition_system
../venv/bin/pip install -r requirements.txt
../venv/bin/python tests/test_pipeline.py              # or: python -m pytest tests
../venv/bin/python scripts/benchmark_denoising.py      # ~45 s
../venv/bin/python scripts/analyze_session.py data/custom_radial/subject_custom_1000hz.csv
```

Recording with the band (after flashing `firmware/nadi_tri_sensor`):

```bash
../venv/bin/python scripts/record_session.py --subject S01 --protocol rest --monitor
../venv/bin/python scripts/record_session.py --subject S01 --protocol sweep --duration 90
../venv/bin/python scripts/analyze_session.py data/sessions/S01_sweep_*.csv --sweep
```

In code:

```python
from pulse_core import load_session, run_pipeline
res = run_pipeline(load_session("data/sessions/S01_rest_....csv"))
for ch in res.channels:
    print(ch.name, ch.acceptance, ch.features)
```

## Pipeline

```text
raw IR counts ─► AC/DC (%) ─► HR track f0(t) ─► FSR force cancel ─► HP 0.5·f0 ─► wavelet ─► LP 8·f0
                                   │                                                        │
                                   └──────────── beat detection (spacing from f0) ◄─────────┘
                                                        │
                    per-subject template + robust z-scores + cross-channel check
                                                        │
                              accepted beats ─► ensemble beat ─► features
```

## Benchmark (20 synthetic subjects + 6 VitalDB pulses, two noise levels)

| metric (mean) | v1 fixed band-pass | adaptive (full) |
|---|---|---|
| accepted beats that are actually clean | 83–92 % | **99.8–100 %** |
| HR error, synthetic moderate / heavy | 4.0 / 5.6 bpm | **1.6 / 2.9 bpm** |
| worst-decile ensemble-beat correlation | 0.92–0.97 | **0.97–0.99** |
| rise-time error, heavy noise | 22–27 ms | **15–16 ms** |
| beats kept (yield) | ~100 % | 50–74 % |

Full tables are written to `results/benchmark_summary_*.csv`. Details,
limitations and real-data findings are in [RESEARCH.md](RESEARCH.md).
