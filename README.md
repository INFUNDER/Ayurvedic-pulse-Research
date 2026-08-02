# Ayurvedic-Pulse-Research

This repository contains a modular, signal-quality-aware research platform for extracting physiological features from pulse signals (e.g., PPG, pressure sensors). It is specifically designed to overcome the challenges of wearable artifacts by dynamically routing signals based on a composite Signal Quality Index (SQI) and providing robust feature extraction tailored for downstream machine learning tasks, such as Ayurvedic Dosha prediction.

## Features
- **Signal-Quality-Aware Routing**: Evaluates Spectral Power Ratio (SPR), Kurtosis, and Autocorrelation (ACF) to dynamically route signals.
- **Dual Detectors**: 
  - `ClinicalDetector`: High-precision, Savitzky-Golay smoothed pipeline for pristine signals.
  - `WearableDetector`: Adaptive dual-moving-average pipeline with local SNR rejection for noisy, wearable data.
- **Robust Feature Extraction**: Extracts time-domain, frequency-domain, and non-linear morphological features.
- **Failure Taxonomy Tracker**: Explicitly logs failure modes (e.g., Missing Fiducials, Low SNR, Numerical Instability) for downstream analysis.
- **Real-Time Ayurvedic Dosha Pipeline**: 
  - Live processing of Vata and Pitta sensors (MAX30102) with spatial resolution maintained.
  - 60-second dynamic baseline calibration to adapt to physiological variance.
  - Continuous 10-second rolling window classification mapping scientific hemodynamic features (e.g. Augmentation Index, Stiffness Index) to traditional parameters (Vega, Gati, Tala, Bala, Kathinya, Akruti).

## Documentation
- [Dosha Classification Walkthrough](docs/dosha_classification/walkthrough.md)

## Version
- Current: v1.1 (Dual-Sensor Nadi Pariksha Architecture)
