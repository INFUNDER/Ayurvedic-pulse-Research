# Dosha Classification Walkthrough

We have successfully implemented the Ayurvedic Dosha classification pipeline based on your extracted PPG features.

## Changes Made
1. **Feature Extraction (`feature_extraction.py`)**
   - Added robust detection of **P (Systolic Peak)**, **V (Dicrotic Notch / Valley)**, and **T (Tidal Wave Peak)**.
   - Extracted amplitudes $h_1, h_2, h_3$ relative to the pulse onset.
   - Accurately computed the **Augmentation Index (AI)** using $AI = \frac{h_1 - h_3}{h_1}$.
   - Computed the **Stiffness Index (SI)** utilizing the peak timing of the dicrotic segment relative to the systolic segment.
2. **Ayurveda Mapper (`ayurveda_mapper.py`)**
   - Implemented mapping logic mapping the above scientific features to **Vega, Gati, Tala, Bala, Kathinya, and Akruti**.
   - Introduced a dynamic scoring system that assigns points to Vata, Pitta, and Kapha across the 6 features to determine the **Dominant Dosha**.
3. **Test Validation (`test_custom_ayurveda.py`)**
   - Refactored the final output terminal print statement to read seamlessly, displaying the individual feature Dosha assignments as well as the total scores.

## Phase 2: Dynamic Baseline Calibration
To transition this pipeline into a robust real-time hardware application, we implemented a dedicated 60-second initialization architecture.

### 1. Artifact Rejection & Calibration (`calibration.py`)
- Created `DoshaCalibrator`, which acts as a 60-second startup buffer.
- Integrated `calculate_sqi` (Signal Quality Index) to automatically **drop any segmented cycles affected by motion artifacts**, preventing them from skewing the user's baseline.
- Calculates and stores a median and standard deviation ($\pm 1 \sigma$) for both $h_1$ (Akruti) and $AI$ (Bala).

### 2. Dynamic Thresholding (`ayurveda_mapper.py`)
- Updated the mapper to accept the `baseline` and a tunable `sigma_multiplier=1.0`.
- Instead of using hardcoded numbers, the classifier now mathematically normalizes hardware shifts and biological variance:
  - **Akruti/Bala = Kapha (Average)**: Falls within $\pm 1 \sigma$ of the user's baseline.
  - **Akruti/Bala = Pitta (High)**: Rises $> 1 \sigma$ above baseline.
  - **Akruti/Bala = Vata (Low)**: Drops $< -1 \sigma$ below baseline.

### 3. Orchestration Script (`realtime_pipeline.py`)
- Separated concerns by keeping `acquire_data.py` strictly as a raw CSV logger.
- Created `realtime_pipeline.py`, a dedicated inference script that listens to a data stream, forces the 60-second calibration, and then enters a continuous 10-second rolling-window classification mode.
- Includes a `--simulate` flag to seamlessly inject existing CSV datasets (like `subject_custom_1000hz.csv`) for off-device testing.

## Phase 3: True Tri-Sensor Nadi Pariksha Architecture
To respect the spatial significance of the Ayurvedic method, the pipeline was upgraded to natively support parallel processing of all three traditional pulse positions: Index finger (Vata), Middle finger (Pitta), and Ring finger (Kapha).

### Firmware Upgrade (`max30102_acquisition.ino`)
- Added initialization and reading logic for `sensorKapha` on Multiplexer Channel 2.
- Updated the serial output format to correctly broadcast all three data streams simultaneously: `Vata:XXXX,Pitta:YYYY,Kapha:ZZZZ`. This guarantees the Python parser can distinguish the physical origin of each raw value.

### Parallel Calibration & Inference (`realtime_pipeline.py`)
- **Triplet Routing:** The serial reader now extracts all three sensor values simultaneously and yields them as `(vata, pitta, kapha)` triplets.
- **Triple Calibrators:** The 60-second initialization now runs three completely independent `DoshaCalibrator` instances simultaneously, generating a dedicated baseline profile for each finger.
- **Distinct Reporting:** During the rolling 10-second inference window, the pipeline evaluates the independent buffers sequentially, generating three distinct reports:
  - `========== VATA SENSOR (Index Finger) 10s Window ==========`
  - `========== PITTA SENSOR (Middle Finger) 10s Window ==========`
  - `========== KAPHA SENSOR (Ring Finger) 10s Window ==========`
- **Preserved Spatial Resolution:** By refusing to mathematically aggregate the sensors, the architecture successfully preserves the spatial diagnostic resolution. A higher-level unified meta-classifier can now easily analyze the distinct combination of these three outputs to render a final holistic diagnosis.
