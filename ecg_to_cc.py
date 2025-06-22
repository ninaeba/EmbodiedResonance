import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
import pretty_midi as pm

# ─── Default Configuration ───────────────────────────────────────────
DEF_BPM       = 75             # Default tempo
CSV_GRID      = 0.1            # Time between CSV samples (in seconds)
SIGNATURE     = "8/8"          # Default time signature
CC_NUMBER     = 1              # Control Change number to use
CC_TRACKS     = ["HR", "SDNN", "RMSSD", "VLF", "LF", "HF", "LF_HF"]
BEAT_PER_GRID = 1 / 32         # Grid resolution for time

# ─── Map value to MIDI CC range 0–127 ───────────────────────────────
def map_to_cc(val: float, vmin: float, vmax: float) -> int:
    norm = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    return int(round(norm * 127))

# ─── Render MIDI CC1 from signal values ─────────────────────────────
def generate_cc_tracks(csv_path: str, out_path: str,
                       grid_sec: float,
                       bpm: int, ppq: int,
                       cells: int, denominator: int,
                       minmax: dict[str, tuple[float, float]]) -> None:

    df = pd.read_csv(csv_path, usecols=lambda c: c.upper() in CC_TRACKS)
    df = df.bfill().ffill().dropna()

    midi = pm.PrettyMIDI(initial_tempo=bpm, resolution=ppq)
    midi.time_signature_changes.append(
        pm.TimeSignature(numerator=cells, denominator=denominator, time=0))

    for name in CC_TRACKS:
        if name not in df.columns:
            continue
        vals = df[name].to_numpy()
        vmin, vmax = minmax[name]

        inst = pm.Instrument(program=0, is_drum=False, name=f"{name}_CC1")

        for i in range(len(vals) - 1):
            beat = i * BEAT_PER_GRID * 4
            t = beat * 60.0 / bpm
            cc_val = map_to_cc(vals[i], vmin, vmax)
            inst.control_changes.append(pm.ControlChange(number=CC_NUMBER, value=cc_val, time=t))

        midi.instruments.append(inst)

    midi.write(out_path)
    print(f"✔ {os.path.basename(out_path)} (CC1 tracks: {', '.join(CC_TRACKS)})")

# ─── Entry Point ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate separate MIDI CC1 tracks from physiological data")
    parser.add_argument("folder", help="Folder with *.csv files")
    parser.add_argument("--tempo", type=int, default=DEF_BPM)
    parser.add_argument("--ppq", type=int, default=480)
    parser.add_argument("--grid", type=float, default=CSV_GRID)
    parser.add_argument("--time-signature", type=str, default=SIGNATURE)
    args = parser.parse_args()

    try:
        num, den = map(int, args.time_signature.split("/"))
    except ValueError:
        raise SystemExit("❌  --time-signature must be in 'num/den' format")

    csv_files = glob.glob(os.path.join(args.folder, "*.csv"))
    if not csv_files:
        raise SystemExit("❌ No CSV files found")

    minmax: dict[str, tuple[float, float]] = {}
    for name in CC_TRACKS:
        vals = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, usecols=lambda c: c.upper() == name)
                if name in df.columns:
                    vals.append(df[name].to_numpy())
            except Exception:
                continue
        if vals:
            all_vals = np.concatenate(vals)
            minmax[name] = (np.nanmin(all_vals), np.nanmax(all_vals))

    for f in csv_files:
        midi_out = os.path.splitext(f)[0] + "_cc_tracks.mid"
        generate_cc_tracks(f, midi_out,
                           grid_sec=args.grid,
                           bpm=args.tempo, ppq=args.ppq,
                           cells=num, denominator=den,
                           minmax=minmax)

if __name__ == "__main__":
    main()
