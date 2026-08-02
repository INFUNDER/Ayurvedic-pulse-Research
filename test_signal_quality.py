import numpy as np
from sensor import load_custom_data

def main():
    sensor_data = load_custom_data("data/custom_radial/subject_custom_1000hz.csv")
    time_array = np.arange(len(sensor_data.signal)) / sensor_data.fs
    valid_mask = time_array > 5.0
    sig = sensor_data.signal[valid_mask]
    
    print(f"Max: {np.max(sig)}")
    print(f"Min: {np.min(sig)}")
    print(f"Mean: {np.mean(sig)}")
    print(f"Std: {np.std(sig)}")

if __name__ == '__main__':
    main()
