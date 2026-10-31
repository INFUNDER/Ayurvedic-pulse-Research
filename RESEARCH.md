# Adaptive noise cancellation for Nadi PPG – research notes & roadmap

*Written 27 Oct 2026 for the 3× MAX30102 + 3× FSR402 wristband. Goal of this
phase (per supervisor): a noise-cancellation algorithm that adapts to each
person, built and validated **before** any dosha labels exist.*

---

## 1. Why a fixed threshold fails, and the rule that replaces it

Every number a fixed threshold looks at (raw ADC counts, peak height, h1
"amplitude" of 5000 or 20000, FSR windows of 100–350) mixes the physiology with
things that have nothing to do with dosha:

| what changes the raw number | typical size |
|---|---|
| skin tone / melanin, tissue depth over the artery | 2–10× |
| LED current, sensor-to-skin coupling | 2–5× |
| hold-down pressure (amplitude rises then falls with force) | up to ~5× |
| heart rate (sets where the useful frequencies are) | 45–110 bpm between people |

**Rule used throughout the new pipeline:** *only threshold quantities that are
already normalised to the person or the recording.*

- amplitude → AC/DC perfusion signal in % (removes gain, skin, LED)
- frequency band → multiples of the person's own tracked heart rate
- beat shape → correlation with the person's *own* running template
- "too big / too small" → robust z-score (median/MAD) within the recording
- noise level → estimated from the data itself (wavelet MAD)
- pressure targets → measured on the person (pressure sweep), not fixed

The few remaining constants (template correlation ≥ 0.85–0.92, k = 3.5 robust
SDs, 8 harmonics) are dimensionless and physiology-independent, so they
transfer between people.

## 2. What the literature does (and what fits this hardware)

| family | idea | fit here |
|---|---|---|
| **Reference-based adaptive filtering** (NLMS/RLS; e.g. Ram et al. 2012) | a second sensor that sees the noise but not the pulse drives an adaptive canceller | FSR behind each sensor = per-channel contact-force reference. Catch: an FSR on the artery *also* sees the pressure pulse (see §4) |
| **Spectral HR tracking** (TROIKA / JOSS, Zhang 2015) | decompose → sparse spectrum → track HR peak over time | used: 3 sites on one artery share the cardiac peak, so channel fusion + Viterbi tracking |
| **Data-driven wavelet shrinkage** (Donoho & Johnstone 1994; BayesShrink, Chang et al. 2000) | threshold set from noise estimated in the signal itself | used, per recording |
| **Adaptive template SQI** (Li & Clifford 2012; Orphanidou 2015; Lim et al. 2018) | compare each beat to a template built from the same signal | used; the core of the "dynamic" part |
| **Contact-pressure optimisation** (Teng & Zhang 2004; review PMC7570982) | AC amplitude peaks near transmural pressure ≈ 0; optimum differs per person/site | used for per-person calibration and as a candidate Kathinya feature |
| **Self-supervised denoising** (SPEAR, Jain et al. 2024) | autoencoder trained on clean PPG reconstructs corrupted segments | next phase, once enough clean wristband data exists |
| **Foundation-model embeddings** (PaPaGei, 2024/25; inter-subject variability paper in `literature/`) | pretrained PPG encoders, fine-tune with few labels | for the classification phase |

## 3. The architecture implemented (`pulse_core/`)

1. **AC/DC normalisation** (`preprocess.to_pulsatile`): inverted IR divided by
   its own DC gives a dimensionless perfusion signal.
2. **Multi-channel HR tracking** (`hr_tracker`): 8 s windows, robust clipping,
   harmonic summation, fusion of the 3 channels weighted by spectral
   peakiness, Viterbi path with a limit on HR change.
3. **FSR contact-force cancellation** (`preprocess.cancel_contact_force`):
   blockwise (8 s, overlapping) least squares regression of the PPG onto the
   FSR force, both low-passed below 0.6 × HR. The coefficient re-adapts every
   window.
4. **HR-tracking filter + wavelet** (`adaptive_filter`): high-pass at
   0.5·f0(t), BayesShrink wavelet, low-pass at 8·f0(t), all zero-phase.
5. **HR-informed beat detection** (`beats`): local amplitude envelope and
   expected IBI instead of global thresholds.
6. **Subject-adaptive beat quality** (`quality`): template bootstrapped from
   the most self-similar beats and updated with accepted beats only. Each
   beat is rejected for: dropout, implausible IBI (<0.6× or >1.6× expected:
   catches missed/extra beats but keeps genuine irregularity, which is the
   Tala/Vata signal), shape, amplitude outlier, contact-force change, or
   absence on the other sensors.
7. **Ensemble-beat features** (`features`): median accepted beat → rise time,
   crest-time ratio, notch timing/height, reflection index, ΔT (for SI),
   systolic area ratio, APG b/a…e/a and ageing index; HRV from accepted
   beats only; perfusion index.
8. **Pressure-sweep calibration** (`pressure`): amplitude-vs-force curve per
   sensor → F_light / F_opt / F_deep and curve width.

## 4. Findings while building it (worth telling your professor)

1. **Common-mode subtraction across the 3 sensors removes the pulse.** The old
   `differential_engine.py` subtracted 0.7 × the 3-channel mean. In your own
   recordings the band-passed channels correlate 0.5–0.99, because the
   dominant common signal *is* the arterial pulse. Fixed: channels are never
   averaged or subtracted, only cross-checked beat by beat.
2. **The FSR is a contact-force reference *and* a pressure-pulse sensor.**
   Used naively as an NLMS reference it cancels the PPG pulse. Three things
   were needed, each confirmed by the benchmark/tests:
   - 4th-order low-pass of the reference below 0.6 × HR;
   - no dF/dt regressor (it amplifies the leaked pulse): clean-beat precision
     fell from 0.998 to 0.925 with it;
   - fit the coefficient on the low-passed PPG. Without this, with a steady
     strap, in-band pulse correlation after cancellation was only 0.6–0.8;
     with it, ≥ 0.999.
3. **In simulation, FSR regression adds little on top of the HR-tracking
   high-pass**, because force artifacts are mostly below the heart rate. The
   FSR's clearest value is (a) contact-change gating of beats and (b)
   per-person pressure calibration. Whether regression helps more on real
   motion (tapping, wrist flexion) must be checked on recordings with FSR
   data.
4. **Your existing 3-sensor recordings are not usable for morphology yet; the
   single-sensor ones are.** Label-free quality from `analyze_session.py`:

   | recording | beats accepted | median template corr | HR-track confidence |
   |---|---|---|---|
   | single sensor, July (100 Hz and 1 kHz) | **95–100 %** | **0.995–0.998** | 0.44 |
   | 3-sensor sessions, Aug–Sep (9 files) | 2–26 % | 0.23–0.80 | 0.13–0.21 |

   Same chip, same pipeline, so the gap is acquisition, not processing.
   Likely causes:
   - LED current was forced down to 0x0A (~2 mA) to avoid brown-outs;
   - no fixed hold-down (no enclosure yet);
   - early sessions at 23–25 Hz.

   The new firmware adds per-sensor auto-gain (capped) and the enclosure
   fixes the mechanics. Re-record before drawing any conclusion.
5. **PWV ("Vega") cannot be measured between the three sites.** At 10–12 mm
   spacing and 5–10 m/s, the transit delay is ~1–2 ms, below one sample at
   100 Hz and below the jitter. Use HR for Vega, or add a second site
   (e.g. a finger) for PTT.
6. **Known limitation of the HR tracker:** with very low perfusion (< 0.5 %)
   and dense artifacts (several per 10 s), it can lock to a sub-harmonic
   (one VitalDB case under heavy noise). It fails safe: the beat-quality stage
   then rejects most beats, so the output is "few beats", not wrong features.
   Record ≥ 60–90 s at rest. Tried and rejected: Viterbi tuning, spectral
   whitening, absorbance-domain cancellation before tracking (no gain or worse).

## 5. How it was validated without labels

- **Known-truth benchmark** (`scripts/benchmark_denoising.py`):
  - 20 synthetic subjects with different HR, HRV, reflection timing, notch
    depth, perfusion and optimal force, plus 6 real clean VitalDB pulses;
  - realistic noise injected: respiration and drift, contact-force events
    that change DC and gain through an oscillometric curve, motion bursts,
    sensor noise, I2C dropouts;
  - each step of the pipeline added in turn and scored.

  | (mean over subjects) | v1 fixed | full adaptive |
  |---|---|---|
  | accepted beats that are truly clean | 83–92 % | **99.8–100 %** |
  | HR MAE synthetic moderate / heavy | 4.0 / 5.6 bpm | **1.6 / 2.9** |
  | worst-decile ensemble-beat corr | 0.92–0.97 | **0.97–0.99** |
  | rise-time error, heavy noise | 22–27 ms | **15–16 ms** |
  | yield (beats kept) | ~100 % | 50–74 % |

  The trade-off is deliberate: v1 keeps every beat, including 10–17 %
  corrupted ones that then contaminate the features.
- **Repeatability on real data:** CV of features across 30 s windows. On the
  good single-sensor recording, HR CV 2.6 %, crest-time ratio 2.3 %,
  perfusion 6.5 %, notch height 12 %. A trait like Prakriti can only be
  predicted from features that are stable within a person, so this
  (and later test–retest ICC across days) is the gate before any classifier.

## 6. Roadmap

**Phase 1 – now, while the enclosure prints**
- [x] adaptive denoiser + label-free benchmark (this repo)
- [ ] flash `firmware/nadi_tri_sensor`, confirm all 3 sensors + FSRs stream (check `# led_amp=` line)
- [ ] calibrate FSRs with 4–5 known weights → `pressure.fit_fsr_calibration`
- [ ] record yourself: 3 × 2 min rest + 1 pressure sweep, on 3 different days → `analyze_session.py`
- [ ] targets before moving on: ≥ 70 % beats accepted on all 3 channels, HR CV < 5 %, morphology CV < 10 %

**Phase 2 – ground truth (start the ethics application now; it's the longest lead time; you already have CITI training)**
- Labels, in order of availability:
  1. **CCRAS Prakriti Assessment Scale**: one of only two tools meeting 7/9
     validation criteria in a 2025 systematic review (Frontiers in Medicine).
  2. **Blinded vaidya nadi assessment** on a subset (two vaidyas →
     inter-rater κ; this is the ceiling any model can reach).
  3. Prakriti200-style questionnaire scores as weak labels.
- Protocol:
  - morning, ≥ 3 h after food, 5 min seated rest, arm at heart level;
  - pressure sweep, then light / optimal / deep holds;
  - 2–3 sessions per person on different days (test–retest).
- Size: ≥ 30–50 subjects for a pilot. Treat dosha as a **3-part composition**
  (V/P/K %), not a single class.

**Phase 3 – learn without (many) labels**
- Test–retest ICC per feature; drop features with ICC < 0.5.
- Unsupervised structure: per-person feature vectors (3 sites × ~15
  features) → clustering / PCA. Check whether the clusters line up with
  questionnaire scores.
- Self-supervised encoder (PaPaGei-style, or SPEAR-style denoising AE)
  pretrained on all your clean beats; fine-tune with the few labels.

**Phase 4 – supervised, small-N**
- Simple models first (regularised regression on the composition, RF);
  leave-one-subject-out CV only. Windows from the same person must never
  be in both train and test; that leak is the most common reason published
  Nadi accuracies look too good.
- Report against the vaidya inter-rater agreement, not against 100 %.

## 7. References

- Fatangare M, Bhingarkar S. Sensor-based Nadi Pariksha review. *J Ayurveda Integr Med* 15 (2024) 100958 (`literature/`).
- Zhang Z, Pi Z, Liu B. TROIKA: a general framework for heart rate monitoring using wrist-type PPG during intensive physical exercise. *IEEE TBME* 2015. [arXiv:1409.5181](https://arxiv.org/abs/1409.5181)
- Zhang Z. Photoplethysmography-based heart rate monitoring in physical activities via joint sparse spectrum reconstruction (JOSS). *IEEE TBME* 2015. [arXiv:1503.00688](https://arxiv.org/pdf/1503.00688v1)
- Ram MR et al. A novel approach for motion artifact reduction in PPG signals based on AS-LMS adaptive filter. *IEEE Trans Instrum Meas* 2012.
- Donoho DL, Johnstone IM. Ideal spatial adaptation by wavelet shrinkage. *Biometrika* 1994.
- Chang SG, Yu B, Vetterli M. Adaptive wavelet thresholding for image denoising and compression (BayesShrink). *IEEE TIP* 2000.
- Li Q, Clifford GD. Dynamic time warping and machine learning for signal quality assessment of pulsatile signals. *Physiol Meas* 2012.
- Orphanidou C et al. Signal-quality indices for the ECG and PPG. *IEEE JBHI* 2015.
- Lim PK et al. Adaptive template matching of photoplethysmogram pulses to detect motion artefact. *Physiol Meas* 2018. [UM record](https://eprints.um.edu.my/21438/)
- Teng XF, Zhang YT. The effect of contacting force on photoplethysmographic signals. *Physiol Meas* 2004.
- Contact pressure in PPG, review: [PMC7570982](https://pmc.ncbi.nlm.nih.gov/articles/PMC7570982)
- Takazawa K et al. Second derivative of the photoplethysmogram and vascular ageing. *Hypertension* 1998. Elgendi M. On the analysis of fingertip PPG signals. *Curr Cardiol Rev* 2012.
- Jain et al. SPEAR: self-supervised PPG denoising. *Harvard Data Science Review* 2024. [link](https://hdsr.mitpress.mit.edu/pub/rzxsdufo) · [arXiv:2307.05339](https://arxiv.org/pdf/2307.05339)
- Robust heart rate detection via multi-site PPG. [arXiv:2412.17538](https://arxiv.org/pdf/2412.17538)
- PaPaGei: open foundation models for optical physiological signals (`literature/`).
- Prakriti200 dataset. [arXiv:2510.06262](https://arxiv.org/pdf/2510.06262)
- Systematic review of Prakriti assessment tools. *Front Med* 2025. [link](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1656249/full)
