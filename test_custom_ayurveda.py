import numpy as np
from sensor import load_custom_data
from beat_detection import WearableDetector
from feature_extraction import extract_all_features
from ayurveda_mapper import map_to_ayurveda
from scipy.signal import butter, sosfiltfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def main():
    print("Loading custom 1000Hz radial data...")
    sensor_data = load_custom_data("data/custom_radial/subject_custom_1000hz.csv")
    fs = sensor_data.fs
    sig = sensor_data.signal
    time_array = np.arange(len(sig)) / fs
    valid_mask = time_array > 5.0
    sig = sig[valid_mask]
    
    # 0.5 - 10.0 Hz bandpass (perfect for preserving notches)
    filtered_sig = bandpass_filter(sig, 0.5, 10.0, fs, order=4)
    
    print("\nDetecting beats using WearableDetector...")
    detector = WearableDetector()
    det_result = detector.detect(filtered_sig, fs)
    
    class DummyResult:
        pass
    res = DummyResult()
    res.systolic_peaks = det_result.systolic_peaks
    res.confidence = 0.90 
    
    features, status = extract_all_features(filtered_sig, fs, res)
    
    if status == "SUCCESS" and features:
        ayur = map_to_ayurveda(features)
        
        print("\n--- AYURVEDIC NADI PARIKSHA REPORT (Ronit's 1000Hz Pulse) ---")
        print(f"Dominant Dosha: {ayur.get('Dominant_Dosha', 'N/A')}")
        scores = ayur.get('Scores', {})
        print(f"Scores: Vata: {scores.get('Vata', 0)}, Pitta: {scores.get('Pitta', 0)}, Kapha: {scores.get('Kapha', 0)}")
        print("-" * 60)
        
        print(f"1. Vega (Pulse Rate)  : {ayur.get('Vega_HeartRate_BPM', np.nan):.2f} BPM -> {ayur.get('Vega_Dosha', 'N/A')}")
        print(f"2. Gati (Movement)    : -> {ayur.get('Gati_Dosha', 'N/A')}")
        print(f"3. Tala (Rhythm)      : CVNN = {ayur.get('Tala_CVNN', np.nan):.2f}% -> {ayur.get('Tala_Dosha', 'N/A')}")
        print(f"4. Bala (Force)       : AI = {ayur.get('Bala_Augmentation_Index', np.nan):.2f} -> {ayur.get('Bala_Dosha', 'N/A')}")
        print(f"5. Kathinya (Stiffness): SI = {ayur.get('Kathinya_Stiffness_Index', np.nan):.2f} m/s -> {ayur.get('Kathinya_Dosha', 'N/A')}")
        print(f"6. Akruti (Volume)    : h1 = {ayur.get('Akruti_h1', np.nan):.2f} -> {ayur.get('Akruti_Dosha', 'N/A')}")
        print("-" * 60)
        
        print("\nAll Extracted Scientific Features:")
        for k, v in features.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Feature extraction failed. Status: {status}")

if __name__ == '__main__':
    main()
