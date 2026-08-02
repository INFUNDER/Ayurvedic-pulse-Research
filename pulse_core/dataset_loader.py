import os
import pandas as pd
import numpy as np

DATASET_ROOT = "/Users/ronitmittal/Documents/Projects/Meditation Brain state/data/PPG_BP/Data File"

def load_metadata():
    """Loads the PPG-BP dataset metadata spreadsheet."""
    excel_path = os.path.join(DATASET_ROOT, "PPG-BP dataset.xlsx")
    # The header is actually on the second row (index 1)
    df = pd.read_excel(excel_path, header=1)
    
    # Clean up columns and drop empty rows
    df = df.dropna(subset=['subject_ID'])
    return df

def load_subject_segments(subject_id):
    """
    Loads the 3 segments of 2100 samples (sampled at 1000Hz) for a given subject.
    Returns a list of numpy arrays.
    """
    subject_dir = os.path.join(DATASET_ROOT, "0_subject")
    segments = []
    
    for seg_idx in [1, 2, 3]:
        file_path = os.path.join(subject_dir, f"{int(subject_id)}_{seg_idx}.txt")
        if os.path.exists(file_path):
            try:
                # The file is a single line, tab-separated
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        # Values might be separated by tabs or spaces
                        vals = np.array([float(x) for x in content.split() if x])
                        segments.append(vals)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                
    return segments

if __name__ == "__main__":
    df = load_metadata()
    print(f"Loaded metadata for {len(df)} subjects.")
    segs = load_subject_segments(df.iloc[0]['subject_ID'])
    print(f"Loaded {len(segs)} segments for first subject. Shape of seg 1: {segs[0].shape if segs else 'None'}")
