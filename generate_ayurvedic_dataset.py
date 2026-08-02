import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset_loader import load_metadata, load_subject_segments
from beat_detection import ClinicalDetector
from feature_extraction import extract_all_features
from ayurveda_mapper import map_to_ayurveda

def process_subject(subject_id):
    segments = load_subject_segments(subject_id)
    if not segments:
        return None
        
    fs = 1000.0 # PPG-BP dataset is sampled at 1000 Hz
    detector = ClinicalDetector()
    
    all_ayurvedic_features = []
    
    for sig in segments:
        # Preprocessing: detrend to remove baseline wander
        sig = sig - np.mean(sig)
        
        # 1. Detect Beats
        det_result = detector.detect(sig, fs)
        
        # 2. Extract 30 Scientific Features
        # Using a dummy detection_result struct to pass into extraction
        class DummyResult:
            pass
        res = DummyResult()
        res.systolic_peaks = det_result.systolic_peaks
        res.confidence = det_result.confidence
        
        features, status = extract_all_features(sig, fs, res)
        
        if status == "SUCCESS" and features:
            # 3. Map to Ayurveda
            ayurveda_profile = map_to_ayurveda(features)
            all_ayurvedic_features.append(ayurveda_profile)
            
    if not all_ayurvedic_features:
        return None
        
    # Average the features across the 3 segments
    avg_profile = {}
    keys = all_ayurvedic_features[0].keys()
    
    for k in keys:
        vals = [p[k] for p in all_ayurvedic_features if k in p and p[k] is not None and not np.isnan(p[k])]
        if vals:
            avg_profile[k] = np.mean(vals)
        else:
            avg_profile[k] = np.nan
            
    return avg_profile

def main():
    print("Loading PPG-BP Metadata...")
    df = load_metadata()
    
    # We will build a new dataframe containing the original metadata + Ayurvedic features
    results = []
    
    print(f"Processing {len(df)} subjects...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        subject_id = row['subject_ID']
        
        ayurveda_features = process_subject(subject_id)
        
        if ayurveda_features:
            # Combine dicts
            combined = row.to_dict()
            combined.update(ayurveda_features)
            results.append(combined)
            
    final_df = pd.DataFrame(results)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ayurvedic_ml_dataset.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"\nProcessing Complete!")
    print(f"Successfully extracted Ayurvedic features for {len(final_df)} / {len(df)} subjects.")
    print(f"Dataset saved to: {output_path}")
    print("\nColumns available for Machine Learning:")
    for col in final_df.columns:
        print(f"  - {col}")

if __name__ == "__main__":
    main()
