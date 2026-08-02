# Ayurvedic Nadi Pariksha (Pulse Diagnosis) System

This repository contains a complete hardware and software pipeline designed for real-time Ayurvedic Dosha Classification (Vata, Pitta, Kapha) using optical PPG sensors (MAX30102). 

By extracting advanced hemodynamic features from the human pulse wave, this system mathematically maps scientific cardiovascular metrics to the traditional Ayurvedic parameters described in ancient texts (Vega, Gati, Tala, Bala, Kathinya, and Akruti).

## Repository Structure

To make this codebase approachable for new researchers and developers, the logic is cleanly separated:

```text
pulse_acquisition_system/
├── ayurveda/                   # Nadi Pariksha Ayurvedic Logic (The Core Focus)
│   ├── ayurveda_mapper.py      # Maps hemodynamic features to Vata/Pitta/Kapha
│   └── calibration.py          # 60-second baseline stabilization
├── scripts/                    # Executable Entry Points
│   ├── realtime_pipeline.py    # Main real-time dual-sensor classification script
│   └── ...
├── pulse_core/                 # Underlying DSP & Signal Processing Modules
│   ├── feature_extraction.py   # Extracts P, V, T fiducials, AI, SI, etc.
│   ├── sqi.py                  # Motion artifact rejection
│   └── ...
├── data/                       # Raw input datasets for simulation testing
├── results/                    # Output plots, benchmarks, and logs
├── firmware/                   # Arduino sketches for ESP32 hardware sensors
├── hardware_acquisition/       # Python serial loggers
└── docs/                       # Walkthroughs and documentation
```

## What Is Done So Far (Project Status)

We have successfully built a full end-to-end classification pipeline:

1. **Robust Feature Extraction (`pulse_core`)**
   - Built a high-precision Digital Signal Processing (DSP) pipeline that dynamically filters wearable sensor noise.
   - Accurately detects the Systolic Peak (P), Dicrotic Notch (V), and Tidal Wave (T).
   - Extracts clinical cardiovascular metrics: Augmentation Index (AI), Stiffness Index (SI), and structural amplitudes ($h_1, h_2, h_3$).

2. **Ayurvedic Translation Layer (`ayurveda`)**
   - Implemented a rule-engine that translates scientific metrics into Ayurvedic diagnosis:
     - **Vega (Rate):** Heart rate BPM (Fast = Vata, Medium = Pitta, Slow = Kapha).
     - **Tala (Rhythm):** CVNN variance (Irregular = Vata, Regular = Pitta/Kapha).
     - **Bala (Force):** Augmentation Index (High force = Pitta).
     - **Kathinya (Stiffness):** Stiffness Index (Hard = Vata, Soft = Kapha).
     - **Akruti (Volume):** Wave amplitude $h_1$ relative to baseline.

3. **Dynamic Baseline Calibration (`ayurveda/calibration.py`)**
   - Created a 60-second hardware initialization sequence that measures the unique biological baseline of each subject.
   - Automatically drops noisy beats using a Signal Quality Index (SQI).
   - Generates customized standard deviation thresholds ($\pm 1 \sigma$) so the classification is accurate regardless of skin tone or sensor placement.

4. **True Dual-Sensor Nadi Pariksha Architecture (`scripts/realtime_pipeline.py`)**
   - Configured the ESP32 hardware and Python routing to natively support parallel processing of both the Index Finger (Vata) and Middle Finger (Pitta) sensors.
   - Preserves strict spatial significance by generating two distinct rolling 10-second reports rather than aggregating data.
   - Seamlessly handles live serial data streams from the hardware and includes a `--simulate` flag for offline testing.

## Documentation
- [Dosha Classification Walkthrough](docs/dosha_classification/walkthrough.md) - A detailed breakdown of the mathematical mapping and dual-sensor architecture.

## Version
- Current: v1.1 (Dual-Sensor Nadi Pariksha Architecture)
