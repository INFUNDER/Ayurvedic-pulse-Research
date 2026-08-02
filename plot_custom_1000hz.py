import numpy as np
import matplotlib.pyplot as plt
from sensor import load_custom_data
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
    
    # Generate time array
    time_array = np.arange(len(sig)) / fs
    
    # Reject first 5 seconds to remove transients
    valid_mask = time_array > 5.0
    sig = sig[valid_mask]
    
    # Filter it (0.5 to 10 Hz) using the stable SOS filter
    filtered_sig = bandpass_filter(sig, 0.5, 10.0, fs, order=4)
    
    # Grab a 5-second slice (e.g., from second 5 to 10 of the valid data)
    plot_slice = filtered_sig[5000:10000]
    time_slice = np.arange(len(plot_slice)) / fs
    
    plt.figure(figsize=(12, 5))
    plt.plot(time_slice, plot_slice, color='#ff4b4b', linewidth=1.5, label='Filtered IR Signal')
    
    # Optional: Plot the raw signal lightly in the background to show the noise we removed
    raw_slice = sig[5000:10000]
    # Normalize raw to match filtered scale just for visualization
    raw_slice = (raw_slice - np.mean(raw_slice))
    plt.plot(time_slice, raw_slice, color='gray', alpha=0.3, linewidth=0.5, label='Raw (Noisy) Signal')

    plt.title('Your Custom 1000 Hz PPG Waveform', fontsize=14)
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Amplitude (a.u.)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Zoom in tightly
    plt.xlim(0, 5)
    
    # Save the plot
    save_path = '/Users/ronitmittal/.gemini/antigravity-ide/brain/c9f66cd2-16d7-42d9-ac3e-0de72d15d4c6/validation_1000hz_clean.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    main()
