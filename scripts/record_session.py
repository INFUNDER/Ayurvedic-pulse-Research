"""
Record a session from the wristband (firmware/nadi_tri_sensor) to CSV, with an
optional live quality monitor.

    python scripts/record_session.py                    # auto-detect port, log until Ctrl+C
    python scripts/record_session.py --monitor          # + live HR / beat quality every 2 s
    python scripts/record_session.py --subject S01 --protocol sweep --duration 90

Files go to data/sessions/<subject>_<protocol>_<YYYYmmdd_HHMMSS>.csv in the
v2 format that pulse_core.io.load_session reads.
"""
import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse_core import PipelineState, run_pipeline  # noqa: E402
from pulse_core.io import CHANNELS, from_arrays  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = "t_us,ir_vata,red_vata,ir_pitta,red_pitta,ir_kapha,red_kapha,fsr_vata,fsr_pitta,fsr_kapha"
USB_HINTS = ("usbserial", "usbmodem", "uart", "ch340", "cp210", "slab", "wchusb")


def find_port():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(h in (p.device + " " + (p.description or "")).lower() for h in USB_HINTS):
            return p.device
    return ports[0].device if ports else None


def monitor_line(buf, fs, state):
    arr = np.array(buf, float)  # (N, 10)
    ir = arr[:, [1, 3, 5]].T
    fsr = arr[:, [7, 8, 9]].T
    res = run_pipeline(from_arrays(ir, fs, fsr=fsr, channels=CHANNELS), state=state)
    parts = [f"HR {np.median(res.hr_track.hr_bpm):5.1f}"]
    for ch in res.channels:
        f = ch.features
        parts.append(f"{ch.name[:5]}: ok {100 * ch.acceptance:3.0f}% PI {f.get('perfusion_index_pct', np.nan):4.2f}% "
                     f"F {np.median(fsr[CHANNELS.index(ch.name)]):5.0f}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--subject", default="anon")
    ap.add_argument("--protocol", default="rest", help="e.g. rest, sweep (pressure calibration), posture")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until Ctrl+C")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--window", type=float, default=12.0, help="monitor window, s")
    args = ap.parse_args()

    import serial
    port = args.port or find_port()
    if not port:
        sys.exit("No serial port found - is the ESP32 plugged in?")
    out_dir = os.path.join(ROOT, "data", "sessions")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.subject}_{args.protocol}_{datetime.now():%Y%m%d_%H%M%S}.csv")

    ser = serial.Serial(port, args.baud, timeout=1.0)
    print(f"Connected to {port}. Waiting for the sensor header...")
    fs = 100.0
    buf = deque()
    state = PipelineState()
    n = 0
    t0 = last_print = time.time()
    with open(path, "w") as f:
        f.write(HEADER + "\n")
        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("#"):
                    print(line)
                    if line.startswith("# fs="):
                        fs = float(line.split("=")[1])
                        buf = deque(maxlen=int(args.window * fs))
                    continue
                if line.startswith("t_us"):
                    continue
                parts = line.split(",")
                if len(parts) != 10:
                    continue
                f.write(line + "\n")
                n += 1
                if args.monitor:
                    buf.append([float(p) for p in parts])
                now = time.time()
                if now - last_print >= 2.0:
                    last_print = now
                    status = f"[{now - t0:6.1f}s] {n / (now - t0):5.1f} Hz"
                    if args.monitor and buf.maxlen and len(buf) == buf.maxlen:
                        try:
                            status += " | " + monitor_line(buf, fs, state)
                        except Exception as e:  # never let the monitor kill the recording
                            status += f" | monitor error: {e}"
                    print(status)
                    f.flush()
                if args.duration and now - t0 >= args.duration:
                    break
        except KeyboardInterrupt:
            pass
    ser.close()
    print(f"\nSaved {n} samples to {path}")


if __name__ == "__main__":
    main()
