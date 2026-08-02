import os
import sys
import numpy as np
import scipy.io

# To allow importing rrest_synth from the parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from rrest_synth import gen_ppg_signal, gen_bw_noise, gen_am_noise, gen_fm_noise
except ImportError:
    print("Warning: rrest_synth.py not found in parent directory. Synthetic data generation will fail.")

BIDMC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'bidmc_data.mat')

class SensorData:
    def __init__(self, label, signal, fs, ground_truth_hr=None, reference_ecg=None, reference_ecg_fs=None):
        self.label = label
        self.signal = signal
        self.fs = fs
        self.ground_truth_hr = ground_truth_hr
        self.reference_ecg = reference_ecg
        self.reference_ecg_fs = reference_ecg_fs


def load_synthetic_data(secs=60, hr=80, fs=300, seed=0):
    """
    Generates synthetic PPG data with noise mimicking real sensors.
    """
    rng = np.random.default_rng(seed)
    t, v = gen_ppg_signal(secs=secs, hr=hr, fs=fs, amplitude=1)
    
    # Add Frequency, Amplitude, and Baseline Wander noise
    t_fm, v_fm = gen_fm_noise(t, v, fs=0.03, amplitude=0.03)
    t_am, v_am = gen_am_noise(t_fm, v_fm, fs=0.15, amplitude=0.15)
    t_bw, v_bw = gen_bw_noise(t_am, v_am, fs=0.05, amplitude=0.05 * np.ptp(v_am))
    
    # Add Gaussian white noise
    v_noisy = v_bw + rng.normal(0, 0.01 * np.ptp(v_bw), len(v_bw))
    
    return SensorData(label='Synthetic PPG', signal=v_noisy, fs=fs, ground_truth_hr=hr)


def load_bidmc_data(mat_path=BIDMC_PATH, subject_idx=0):
    """
    Loads real PPG data and ground-truth HR from the BIDMC dataset.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"BIDMC dataset not found at {mat_path}")
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = mat['data']
    s = data[subject_idx]
    
    sig = np.array(s.ppg.v, dtype=float)
    fs = int(s.ppg.fs)
    
    # Handle case where ECG reference might be missing
    try:
        ref_ecg = np.array(s.ekg.v, dtype=float)
        ref_ecg_fs = int(s.ekg.fs)
    except AttributeError:
        ref_ecg = None
        ref_ecg_fs = None
    
    # Load reference HR if available
    try:
        ref_hr_v = np.array(s.ref.hr.v, dtype=float)
        mean_hr = np.nanmean(ref_hr_v) 
    except AttributeError:
        mean_hr = None
        
    return SensorData(
        label=f'BIDMC Subject {subject_idx + 1}', 
        signal=sig, 
        fs=fs, 
        ground_truth_hr=mean_hr,
        reference_ecg=ref_ecg,
        reference_ecg_fs=ref_ecg_fs
    )


def load_mimic_data(record_path):
    """
    Loads MIMIC-III matched waveform data using wfdb.
    Assumes the record_path points to a specific segment (e.g., 'data/mimic3wdb_matched/p00/p000020/3544749_0005')
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("Please install wfdb to read MIMIC datasets.")
        
    record = wfdb.rdrecord(record_path)
    sig_names = [n.lower() for n in record.sig_name]
    
    # Find PPG channel (typically 'pleth' or 'ppg')
    ppg_idx = next((i for i, n in enumerate(sig_names) if 'pleth' in n or 'ppg' in n), None)
    if ppg_idx is None:
        raise ValueError(f"No PLETH/PPG channel found in {record_path}")
        
    # Find ECG channel (typically 'ii', 'v', 'ecg')
    ecg_idx = next((i for i, n in enumerate(sig_names) if 'ii' in n or 'v' in n or 'ecg' in n), None)
    
    fs = record.fs
    ppg_sig = record.p_signal[:, ppg_idx]
    ecg_sig = record.p_signal[:, ecg_idx] if ecg_idx is not None else None
    
    return SensorData(
        label=os.path.basename(record_path),
        signal=ppg_sig,
        fs=fs,
        ground_truth_hr=None, # HR might be in a separate numerics record
        reference_ecg=ecg_sig,
        reference_ecg_fs=fs if ecg_sig is not None else None
    )


def load_wesad_data(pkl_path):
    """
    Loads a WESAD subject pickle file.
    """
    import pickle
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        
    # WESAD stores BVP (PPG) at 64Hz from Empatica E4 wristband
    bvp_sig = d['signal']['wrist']['BVP'].flatten()
    bvp_fs = 64
    
    # ECG from chest sensor at 700Hz
    ecg_sig = d['signal']['chest']['ECG'].flatten()
    ecg_fs = 700
    
    subject_id = os.path.basename(pkl_path).split('.')[0]
    
    return SensorData(
        label=f'WESAD {subject_id}',
        signal=bvp_sig,
        fs=bvp_fs,
        ground_truth_hr=None,
        reference_ecg=ecg_sig,
        reference_ecg_fs=ecg_fs
    )


def load_dalia_data(pkl_path):
    """
    Loads a PPG-DaLiA subject pickle file.
    """
    import pickle
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        
    # PPG-DaLiA: keys are bytes if loaded this way, but let's try standard decoded or bytes
    # Depending on python version and how it was pickled
    try:
        bvp_sig = d['signal']['wrist']['BVP'].flatten()
        ecg_sig = d['signal']['chest']['ECG'].flatten()
    except KeyError:
        bvp_sig = d[b'signal'][b'wrist'][b'BVP'].flatten()
        ecg_sig = d[b'signal'][b'chest'][b'ECG'].flatten()
        
    bvp_fs = 64
    ecg_fs = 700
    
    subject_id = os.path.basename(pkl_path).split('.')[0]
    
    return SensorData(
        label=f'PPG-DaLiA {subject_id}',
        signal=bvp_sig,
        fs=bvp_fs,
        ground_truth_hr=None, # They have HR labels but we will calculate from ECG
        reference_ecg=ecg_sig,
        reference_ecg_fs=ecg_fs
    )

def load_custom_data(csv_path):
    import pandas as pd
    import numpy as np
    import os
    
    df = pd.read_csv(csv_path)
    
    # Calculate empirical sampling rate from timestamps (in milliseconds)
    # Using the median diff is robust to occasional jitter in serial comms
    ts = df['Timestamp_ms'].values
    median_dt_ms = np.median(np.diff(ts))
    fs = 1000.0 / median_dt_ms if median_dt_ms > 0 else 100.0
    
    # The MAX30102 has Red and IR. IR is typically slightly cleaner for pulse on the wrist/finger,
    # but we can try Red first or invert it if necessary. Let's use Red for now, but invert it if 
    # it resembles a transmittance rather than reflectance (usually MAX30102 requires inversion 
    # for standard peak detection since more blood = less light = lower ADC value).
    # We will simply pass the raw IR signal (often better for heart rate) and invert it.
    signal = -df['IR'].values 
    
    subject_id = os.path.basename(csv_path).replace('.csv', '')
    
    return SensorData(
        signal=signal,
        fs=fs,
        reference_ecg=None,
        reference_ecg_fs=None,
        label=subject_id
    )
