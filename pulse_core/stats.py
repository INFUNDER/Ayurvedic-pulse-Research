import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import scipy.stats as st

def run_stats():
    try:
        df = pd.read_csv('benchmark_results_comprehensive.csv')
    except Exception as e:
        print("Run benchmark first.")
        return

    # Filter to only the automatic quality-aware detector
    auto_df = df[df['Detector'] == 'ours_auto']
    
    print("\n--- Feature Extraction Success & Failure Modes ---")
    status_cols = [c for c in auto_df.columns if c.startswith('Status_')]
    if status_cols:
        rows = []
        for col in status_cols:
            feat_name = col.replace('Status_', '')
            counts = auto_df[col].value_counts(normalize=True) * 100
            
            success_pct = counts.get('Success', 0.0)
            rejected_pct = counts.get('Signal Rejected', 0.0)
            low_conf_pct = counts.get('Low Confidence', 0.0)
            fiducial_pct = counts.get('Missing Fiducials', 0.0)
            instability_pct = counts.get('Numerical Instability', 0.0)
            
            rows.append({
                'Feature': feat_name,
                'Success %': round(success_pct, 1),
                'Signal Rejected %': round(rejected_pct, 1),
                'Low Confidence %': round(low_conf_pct, 1),
                'Missing Fiducials %': round(fiducial_pct, 1),
                'Num. Instability %': round(instability_pct, 1)
            })
            
        success_df = pd.DataFrame(rows)
        print(success_df.to_markdown(index=False))
        
    print("\n--- Signal Quality Confusion Matrix (Composite SQI) ---")
    if 'SQI_Class' in auto_df.columns:
        auto_df['Condition_Type'] = auto_df['Condition'].apply(
            lambda x: 'Clean' if x == 'Clean' else ('Downsampled' if 'Downsampled' in x else x)
        )
        confusion = pd.crosstab(auto_df['Condition_Type'], auto_df['SQI_Class'], normalize='index') * 100
        print(confusion.round(1).to_markdown())

    print("\n--- Detector Routing ---")
    if 'Detector_Used' in auto_df.columns:
        route_counts = auto_df['Detector_Used'].value_counts(normalize=True) * 100
        print(route_counts.round(1).to_markdown())

    print("\n--- Failure Taxonomy (F1 < 0.50 or Rejected) ---")
    if 'Failure_Type' in auto_df.columns:
        failures = auto_df[auto_df['Failure_Type'] != 'None']
        if not failures.empty:
            fail_counts = failures['Failure_Type'].value_counts(normalize=True) * 100
            print(fail_counts.round(1).to_markdown())
        else:
            print("No catastrophic failures recorded.")
            
    print("\n--- 95% Confidence Intervals for F1 Score (Clean) ---")
    clean_df = df[df['Condition'] == 'Clean']
    for dataset in clean_df['Dataset'].unique():
        data = clean_df[(clean_df['Dataset'] == dataset) & (clean_df['Detector'] == 'ours_auto')]['F1'].dropna()
        if len(data) > 1:
            mean = np.mean(data)
            sem = st.sem(data)
            ci = sem * st.t.ppf((1 + 0.95) / 2., len(data)-1)
            print(f"{dataset} (Our Auto): {mean:.4f} ± {ci:.4f}")
        elif len(data) == 1:
            print(f"{dataset} (Our Auto): {np.mean(data):.4f} (Not enough samples for CI)")
            
    print("\n--- Wilcoxon Signed-Rank Test (Clean Data) ---")
    ours_f1 = clean_df[clean_df['Detector'] == 'ours_auto']['F1'].values
    heartpy_f1 = clean_df[clean_df['Detector'] == 'heartpy']['F1'].values
    elgendi_f1 = clean_df[clean_df['Detector'] == 'elgendi']['F1'].values
    
    if len(ours_f1) == len(heartpy_f1) and len(ours_f1) > 0:
        try:
            stat, p = wilcoxon(ours_f1, heartpy_f1)
            print(f"Ours vs HeartPy: p-value = {p:.4e}")
        except ValueError:
            pass
            
    if len(ours_f1) == len(elgendi_f1) and len(ours_f1) > 0:
        try:
            stat, p = wilcoxon(ours_f1, elgendi_f1)
            print(f"Ours vs Elgendi: p-value = {p:.4e}")
        except ValueError:
            pass

if __name__ == '__main__':
    run_stats()
