import matplotlib.pyplot as plt
from .sensor import load_bidmc_data, load_synthetic_data
from .noise_removal import remove_noise
from .beat_detection import detect_beats
from .pulse_segmentation import segment_pulses
from .feature_extraction import extract_all_features

def run_pipeline(sensor_data, plot=True):
    print(f"Running pipeline on: {sensor_data.label}")
    
    # 1. Noise Removal
    filtered_signal, trim = remove_noise(sensor_data.signal, sensor_data.fs)
    
    # Trim the signal to avoid edge transients
    trimmed_sig = filtered_signal[trim:-trim]
    
    # 2. Beat Detection
    fiducial_points = detect_beats(trimmed_sig, sensor_data.fs)
    
    # 3. Pulse Segmentation
    segments, segment_indices = segment_pulses(trimmed_sig, fiducial_points['onsets'])
    print(f"Detected {len(segments)} individual pulses.")
    
    # 4. Feature Extraction
    features = extract_all_features(trimmed_sig, sensor_data.fs, fiducial_points)
    
    print("\n--- Extracted Feature Library ---")
    print(f"Total features extracted: {len(features)}")
    
    # Print a sample of 20 features to avoid console spam
    for k, v in list(features.items())[:20]:
        print(f"{k}: {v:.4f}")
    if len(features) > 20:
        print(f"... and {len(features)-20} more features.")
        
    # Validation against ground truth
    estimated_hr = features.get('Heart_Rate_BPM', 0)
    if sensor_data.ground_truth_hr is not None:
        mae = abs(estimated_hr - sensor_data.ground_truth_hr)
        print(f"\n--- Validation ---")
        print(f"Estimated HR:    {estimated_hr:.2f} BPM")
        print(f"Ground Truth HR: {sensor_data.ground_truth_hr:.2f} BPM")
        print(f"Mean Absolute Error (MAE): {mae:.2f} BPM")
        
    if plot:
        plt.figure(figsize=(12, 6))
        # Plot only a 5 second window for clarity
        window_size = min(int(5 * sensor_data.fs), len(trimmed_sig))
        t = [i / sensor_data.fs for i in range(window_size)]
        
        plt.plot(t, trimmed_sig[:window_size], label='Filtered PPG', color='blue', alpha=0.6)
        
        # Helper to plot points
        def plot_points(pts_array, color, marker, label, s=50):
            valid_pts = [p for p in pts_array if p < window_size]
            if valid_pts:
                plt.scatter([p/sensor_data.fs for p in valid_pts], 
                            [trimmed_sig[p] for p in valid_pts], 
                            color=color, marker=marker, s=s, label=label, zorder=5)
                            
        plot_points(fiducial_points['systolic_peaks'], 'red', 'x', 'Systolic Peak', 100)
        plot_points(fiducial_points['onsets'], 'green', 'o', 'Onset (Foot)')
        plot_points(fiducial_points['tidal_waves'], 'purple', 's', 'Tidal Wave')
        plot_points(fiducial_points['valleys'], 'brown', 'v', 'Valley')
        plot_points(fiducial_points['dicrotic_peaks'], 'cyan', '^', 'Dicrotic Peak')
        plot_points(fiducial_points['dicrotic_notches'], 'orange', 'D', 'Dicrotic Notch')
        
        plt.title(f"Expanded Pulse Acquisition Pipeline - {sensor_data.label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'validation_plot_{sensor_data.label.replace(" ", "_")}.png')
        print(f"\nSaved validation plot as validation_plot_{sensor_data.label.replace(' ', '_')}.png")

def main():
    try:
        # Validate with real clinical dataset
        bidmc_data = load_bidmc_data(subject_idx=2) # Using subject 3
        run_pipeline(bidmc_data, plot=True)
    except FileNotFoundError as e:
        print(f"BIDMC dataset missing: {e}. Falling back to Synthetic data.")
        synth_data = load_synthetic_data()
        run_pipeline(synth_data, plot=True)

if __name__ == "__main__":
    main()
