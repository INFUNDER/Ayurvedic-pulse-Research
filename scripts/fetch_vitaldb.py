"""
scripts/fetch_vitaldb.py

Downloads clean clinical finger PPG (SNUADC/PLETH, 500 Hz) from the open VitalDB
API and saves it at 250 Hz to data/vitaldb_clinical/. These recordings are the
"real morphology" half of scripts/benchmark_denoising.py: clean pulses onto
which realistic wristband noise is injected.

Note: out-of-range samples are dropped, which joins the pieces on either side,
so the benchmark only uses windows with regular beats (see vitaldb_pulses()).
"""

import gzip
import os
import urllib.request

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vitaldb_clinical")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VitalDB case IDs to fetch (surgical patients under anaesthesia; any IDs with a PLETH track work)
TARGET_CASES = [1, 2, 3, 5, 7, 10]


def get_track_id_map():
    """Fetches track ID mapping from VitalDB API."""
    print("Connecting to VitalDB API track registry...")
    url = "https://api.vitaldb.net/trks"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    res = urllib.request.urlopen(req, timeout=15)
    raw = res.read()
    data = gzip.decompress(raw).decode("utf-8-sig")
    
    pleth_map = {}
    for line in data.strip().split("\n")[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 3 and parts[1] == "SNUADC/PLETH":
            try:
                cid = int(parts[0])
                tid = parts[2]
                pleth_map[cid] = tid
            except ValueError:
                pass
    print(f"Found {len(pleth_map)} total clinical PPG tracks in VitalDB.")
    return pleth_map

def download_and_process_case(cid, tid):
    target_path = os.path.join(OUTPUT_DIR, f"case_{cid:04d}_pleth_250hz.npy")
    if os.path.exists(target_path):
        print(f"Case {cid} already exists at {target_path}, skipping download.")
        return target_path
        
    print(f"Downloading Real Clinical Case {cid} (Track: {tid[:8]}...)...")
    url = f"https://api.vitaldb.net/{tid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    try:
        res = urllib.request.urlopen(req, timeout=30)
        raw = res.read()
        decompressed = gzip.decompress(raw).decode("utf-8")
    except Exception as e:
        print(f"Failed to download Case {cid}: {e}")
        return None
        
    lines = decompressed.strip().split("\n")
    print(f"Parsing {len(lines)} raw 500 Hz samples...")
    
    values = []
    for line in lines[1:]:
        p = line.split(",")
        if len(p) > 1 and p[1].strip():
            try:
                val = float(p[1])
                values.append(val)
            except ValueError:
                pass
                
    values = np.array(values, dtype=np.float32)
    
    # Clean out extreme sensor-off rails (< -50 or > 150)
    valid_mask = (values >= -50.0) & (values <= 150.0)
    values = values[valid_mask]
    
    if len(values) < 5000:
        print(f"Case {cid} has insufficient valid samples, skipping.")
        return None
        
    # Downsample from 500 Hz to 250 Hz (decimation by 2 with 2-tap averaging filter)
    # y[n] = (x[2n] + x[2n+1]) / 2
    truncated_len = (len(values) // 2) * 2
    even = values[:truncated_len:2]
    odd = values[1:truncated_len:2]
    downsampled = (even + odd) * 0.5
    
    # Save as compressed .npy
    np.save(target_path, downsampled)
    duration_min = len(downsampled) / 250.0 / 60.0
    print(f"Successfully saved Case {cid}: {len(downsampled):,} samples ({duration_min:.1f} mins) to {target_path}")
    return target_path

def main():
    print("============================================================")
    print("📥 Starting Real Clinical Data Ingestion (VitalDB Open API)")
    print("============================================================")
    pleth_map = get_track_id_map()
    
    downloaded_files = []
    for cid in TARGET_CASES:
        if cid in pleth_map:
            path = download_and_process_case(cid, pleth_map[cid])
            if path:
                downloaded_files.append(path)
                
    total_samples = sum(len(np.load(f)) for f in downloaded_files)
    print(f"\n============================================================")
    print(f"✅ Ingestion Complete!")
    print(f"📁 Downloaded Cases : {len(downloaded_files)}")
    print(f"⏱️ Total Duration   : {total_samples / 250 / 3600:.2f} hours of continuous clinical waveforms")
    print(f"============================================================")

if __name__ == "__main__":
    main()
