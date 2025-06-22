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

# Time resolution and rhythm settings per signal
VLF_CELL_SEC  = 0.8
LF_CELL_SEC   = 0.4
HF_CELL_SEC   = 0.2
VLF_STEP_BEAT = 1 / 4
LF_STEP_BEAT  = 1 / 8
HF_STEP_BEAT  = 1 / 16
VLF_NOTE_LEN  = 1 / 4 
LF_NOTE_LEN   = 1 / 8
HF_NOTE_LEN   = 1 / 16

# CC Numbers
CC_SIGNAL     = 1
CC_RATIO      = 11

# ─── Musical Scales ──────────────────────────────────────────────────
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]  # C major intervals
MINOR_SCALE = [0, 2, 3, 5, 7, 10, 11] # C minor intervals

# ─── Normalize value into scale-based pitch ─────────────────────────
def normalize_note(val: float, vmin: float, vmax: float, scale: List[int], base_pitch: int) -> int:
    norm = np.clip((val - vmin) / (vmax - vmin), 0, 0.999)
    index = int(norm * len(scale))
    return base_pitch + scale[index]

# ─── Map value to velocity range [80, 120] ───────────────────────────
def map_to_velocity(val: float, vmin: float, vmax: float) -> int:
    norm = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    return int(round(80 + norm * (120 - 80)))

# ─── Convert CSV physiological bands to multiple MIDI tracks ────────
def signals_to_midi(csv_path: str, out_path: str,
                    grid_sec: float,
                    bpm: int, ppq: int,
                    cells: int, denominator: int,
                    vlf_min: float, vlf_max: float,
                    lf_min: float, lf_max: float,
                    hf_min: float, hf_max: float) -> None:

    df = pd.read_csv(csv_path, usecols=lambda c: c.upper() in ["VLF", "LF", "HF", "LF_HF"])
    df = df.bfill().ffill().dropna()

    vlf_vals   = df["VLF"].to_numpy()
    lf_vals    = df["LF"].to_numpy()
    hf_vals    = df["HF"].to_numpy()
    lf_hf_vals = df["LF_HF"].to_numpy()

    midi = pm.PrettyMIDI(initial_tempo=bpm, resolution=ppq)
    midi.time_signature_changes.append(
        pm.TimeSignature(numerator=cells, denominator=denominator, time=0))

    inst_vlf = pm.Instrument(program=0, is_drum=False, name="VLF_Notes")
    inst_lf  = pm.Instrument(program=0, is_drum=False, name="LF_Notes")
    inst_hf  = pm.Instrument(program=0, is_drum=False, name="HF_Notes")

    for name, vals, cell_sec, step_beat, note_len, base_pitch, vmin, vmax, inst in [
        ("VLF", vlf_vals, VLF_CELL_SEC, VLF_STEP_BEAT, VLF_NOTE_LEN, 36, vlf_min, vlf_max, inst_vlf),
        ("LF",  lf_vals,  LF_CELL_SEC,  LF_STEP_BEAT,  LF_NOTE_LEN,  48, lf_min,  lf_max,  inst_lf),
        ("HF",  hf_vals,  HF_CELL_SEC,  HF_STEP_BEAT,  HF_NOTE_LEN,  60, hf_min,  hf_max,  inst_hf)
    ]:
        step = int(round(cell_sec / grid_sec))
        indices = np.arange(0, len(vals), step)

        for count, i in enumerate(indices):
            beat = count * step_beat * 4
            t = beat * 60.0 / bpm
            duration = note_len * 4 * 60.0 / bpm
            scale = MAJOR_SCALE if lf_hf_vals[i] > 2 else MINOR_SCALE
            pitch = normalize_note(vals[i], vmin, vmax, scale, base_pitch)
            velocity = map_to_velocity(vals[i], vmin, vmax)
            inst.notes.append(pm.Note(velocity=velocity, pitch=pitch, start=t, end=t + duration))

            # Add CC1 automation for this signal
            norm = np.clip((vals[i] - vmin) / (vmax - vmin), 0, 1)
            cc_val = int(round(norm * 127))
            inst.control_changes.append(pm.ControlChange(number=CC_SIGNAL, value=cc_val, time=t))

            # Add CC11 automation for LF/HF
            lf_hf_norm = np.clip((lf_hf_vals[i] - np.min(lf_hf_vals)) / (np.max(lf_hf_vals) - np.min(lf_hf_vals)), 0, 1)
            cc_lf_hf_val = int(round(lf_hf_norm * 127))
            inst.control_changes.append(pm.ControlChange(number=CC_RATIO, value=cc_lf_hf_val, time=t))

    midi.instruments.extend([inst_vlf, inst_lf, inst_hf])
    midi.write(out_path)
    print(f"✔ {os.path.basename(out_path)} (notes per track: VLF/LF/HF)")

# ─── Entry Point ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate MIDI notes from physiological signals (VLF, LF, HF)")
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

    vlf_vals, lf_vals, hf_vals = [], [], []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c.upper() in ["VLF", "LF", "HF"])
            if "VLF" in df.columns:
                vlf_vals.append(df["VLF"].to_numpy())
            if "LF" in df.columns:
                lf_vals.append(df["LF"].to_numpy())
            if "HF" in df.columns:
                hf_vals.append(df["HF"].to_numpy())
        except Exception as e:
            print(f"⚠ Error reading {f}: {e}")

    vlf_all = np.concatenate(vlf_vals)
    lf_all = np.concatenate(lf_vals)
    hf_all = np.concatenate(hf_vals)

    vlf_min, vlf_max = np.nanmin(vlf_all), np.nanmax(vlf_all)
    lf_min,  lf_max  = np.nanmin(lf_all),  np.nanmax(lf_all)
    hf_min,  hf_max  = np.nanmin(hf_all),  np.nanmax(hf_all)

    for f in csv_files:
        midi_out = os.path.splitext(f)[0] + "_signals_notes.mid"
        signals_to_midi(f, midi_out,
                        grid_sec=args.grid,
                        bpm=args.tempo, ppq=args.ppq,
                        cells=num, denominator=den,
                        vlf_min=vlf_min, vlf_max=vlf_max,
                        lf_min=lf_min, lf_max=lf_max,
                        hf_min=hf_min, hf_max=hf_max)

if __name__ == "__main__":
    main()
