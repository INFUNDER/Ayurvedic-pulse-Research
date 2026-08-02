import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_bland_altman(data1, data2, title, save_path):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   
    md = np.mean(diff)                     
    sd = np.std(diff, axis=0)              
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.6, edgecolors='k')
    plt.axhline(md, color='red', linestyle='--', label=f'Mean Bias: {md:.2f}')
    plt.axhline(md + 1.96*sd, color='blue', linestyle='--', label=f'+1.96 SD: {md + 1.96*sd:.2f}')
    plt.axhline(md - 1.96*sd, color='blue', linestyle='--', label=f'-1.96 SD: {md - 1.96*sd:.2f}')
    
    plt.title(title)
    plt.xlabel('Mean HR (BPM)')
    plt.ylabel('Difference (Detected HR - True HR) (BPM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_plots():
    if not os.path.exists('benchmark_results_comprehensive.csv') or not os.path.exists('benchmark_errors.csv'):
        print("Run benchmark first.")
        return
        
    df = pd.read_csv('benchmark_results_comprehensive.csv')
    err_df = pd.read_csv('benchmark_errors.csv')
    
    # 1. Bland-Altman Plots
    clean_ours = df[(df['Condition'] == 'Clean') & (df['Detector'] == 'ours')].dropna(subset=['HR', 'True_HR'])
    if not clean_ours.empty:
        plot_bland_altman(
            clean_ours['HR'], 
            clean_ours['True_HR'], 
            'Bland-Altman Plot: Our Algorithm vs ECG Reference (Clean Data)',
            'bland_altman_clean.png'
        )
        print("Saved bland_altman_clean.png")
        
    # 2. Cumulative Distribution Function (CDF) of Timing Errors (Multi-panel)
    datasets = err_df['Dataset'].unique()
    num_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, num_datasets, figsize=(15, 5), sharey=True)
    if num_datasets == 1:
        axes = [axes]
        
    for ax, dataset in zip(axes, datasets):
        dataset_errors = err_df[(err_df['Dataset'] == dataset) & (err_df['Condition'] == 'Clean')]['Error_ms'].dropna()
        if not dataset_errors.empty:
            abs_errors = np.abs(dataset_errors)
            sorted_errors = np.sort(abs_errors)
            yvals = np.arange(len(sorted_errors)) / float(len(sorted_errors) - 1)
            ax.plot(sorted_errors, yvals, label=dataset, linewidth=2, color='darkorange')
            
        ax.set_title(f'{dataset} Timing Errors')
        ax.set_xlabel('Absolute Error (ms)')
        ax.set_xlim(0, 150)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    axes[0].set_ylabel('Cumulative Probability')
    plt.tight_layout()
    plt.savefig('error_cdf_clean.png', dpi=300)
    plt.close()
    print("Saved error_cdf_clean.png")

if __name__ == '__main__':
    generate_plots()
