import numpy as np
import scipy.signal
from sensor import load_custom_data

sensor_data = load_custom_data("data/custom_radial/subject_custom_1000hz.csv")
sig = sensor_data.signal
fs = sensor_data.fs
time_array = np.arange(len(sig)) / fs
valid = time_array > 5.0
sig = sig[valid]

from scipy.signal import butter, filtfilt
b, a = butter(4, [0.5/(fs/2), 10.0/(fs/2)], btype='band')
filt = filtfilt(b, a, sig)

print(f"Raw std: {np.std(sig):.2f}")
print(f"Filtered std: {np.std(filt):.2f}")
print(f"Filtered max: {np.max(filt):.2f}")
print(f"Filtered min: {np.min(filt):.2f}")

