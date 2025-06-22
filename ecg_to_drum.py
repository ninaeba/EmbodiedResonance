import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pretty_midi as pm
import math

# ─── Default Configuration ───────────────────────────────────────────
STEP_BEAT     = 1 / 32           # Relative beat step used in beat-to-second conversion
NOTE_LENGTH   = 1 / 32           # Length of each drum note (in beats)
DEF_BPM       = 75              # Default tempo in BPM
CSV_GRID      = 0.1             # CSV sample interval in seconds
SEC_PER_CELL  = 0.1             # Logical duration of one cell in seconds
SIGNATURE     = "16/16"         # Default time signature
CC_SDNN       = 11              # CC number for SDNN modulation
CC_RMSSD      = 1               # CC number for RMSSD modulation

# Map index to cell position to define a musical spatialization scheme
CELL_MAP = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
# CELL_MAP = [0, 4, 2, 6, 3, 7, 1, 5]  # for 8 cells patern

# ─── Map HR ranges to percussive triggers ─────────────────────────────
def generate_rules(hr_min: float, hr_max: float, cells: int) -> List[Tuple[float, int, int]]:
    thresholds = np.linspace(hr_min, hr_max, num=cells + 1)[1:]
    rules = [(thr, CELL_MAP[i], 36 + i) for i, thr in enumerate(thresholds)]
    rules[-1] = (np.inf, rules[-1][1], rules[-1][2])  # Last rule handles all above
    return rules

# ─── Main conversion from CSV to MIDI ─────────────────────────────────
def csv_to_midi(csv_path: str, out_path: str, *,
                bpm: int, ppq: int,
                seconds_per_cell: float,
                grid_sec: float,
                cells: int,
                denominator: int,
                rules: List[Tuple[float, int, int]],
                sdnn_min: float, sdnn_max: float,
                rmssd_min: float, rmssd_max: float) -> None:

    df = pd.read_csv(csv_path, usecols=lambda c: c.upper().startswith(("HR", "SDNN", "RMSSD")))
    df = df.bfill().ffill().dropna()

    hr_vals     = df.iloc[:, 0].to_numpy()
    sdnn_vals   = df.iloc[:, 1].to_numpy()
    rmssd_vals  = df.iloc[:, 2].to_numpy()

    drum = pm.Instrument(program=0, is_drum=True, name="HR_Drums")
    beat_per_grid = grid_sec / seconds_per_cell * STEP_BEAT * 4  # May be revised later

    for i in range(len(sdnn_vals)):
        beat = i * beat_per_grid
        t = beat * 60.0 / bpm

        sdnn_norm  = np.clip((sdnn_vals[i] - sdnn_min) / (sdnn_max - sdnn_min), 0, 1)
        rmssd_norm = np.clip((rmssd_vals[i] - rmssd_min) / (rmssd_max - rmssd_min), 0, 1)

        drum.control_changes.append(pm.ControlChange(number=CC_SDNN, value=int(round(sdnn_norm * 127)), time=t))
        drum.control_changes.append(pm.ControlChange(number=CC_RMSSD, value=int(round(rmssd_norm * 127)), time=t))

    # Build cumulative trigger pattern
    CUM_PATTERNS: List[List[Tuple[int, int]]] = []
    acc: List[Tuple[int, int]] = []
    for thr, cell, note in rules:
        acc.append((cell, note))
        CUM_PATTERNS.append(list(acc))

    if hr_vals.size == 0:
        print(f"⚠ {os.path.basename(csv_path)}: no HR column found – skipped")
        return

    times = np.arange(len(hr_vals)) * grid_sec
    total_duration = len(hr_vals) * grid_sec
    max_bars = int(total_duration // (seconds_per_cell * cells))

    if max_bars == 0:
        print(f"⚠ {os.path.basename(csv_path)}: shorter than one bar – skipped")
        return

    midi = pm.PrettyMIDI(initial_tempo=bpm, resolution=ppq)
    midi.time_signature_changes.append(
        pm.TimeSignature(numerator=cells, denominator=denominator, time=0))
    beat2sec = lambda beat: beat * 60.0 / bpm

    for bar in range(max_bars):
        for cell in range(cells):
            t_sample = (bar * cells + cell) * seconds_per_cell
            if t_sample > times[-1]:
                continue

            hr = np.interp(t_sample, times, hr_vals)
            zone = next((i for i, (thr, *_n) in enumerate(rules) if hr <= thr), len(rules) - 1)
            vel = int(round(hr))

            for z_cell, note in CUM_PATTERNS[zone]:
                if z_cell != cell:
                    continue
                beat = (bar * cells + cell) * STEP_BEAT * 4
                t0 = beat2sec(beat)
                drum.notes.append(pm.Note(velocity=vel, pitch=note,
                                          start=t0, end=t0 + beat2sec(NOTE_LENGTH * 4)))

    midi.instruments.append(drum)
    midi.write(out_path)
    print(f"✔ {os.path.basename(out_path)}  ({max_bars} bars, {cells}/{denominator})")

# ─── Entry point ──────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Convert HR/SDNN/RMSSD to MIDI drum patterns")
    ap.add_argument("folder", help="Folder containing *.csv files with HR data")
    ap.add_argument("--tempo", type=int, default=DEF_BPM)
    ap.add_argument("--ppq", type=int, default=480)
    ap.add_argument("--seconds-per-cell", type=float, default=SEC_PER_CELL,
                    help="Time window (sec) used to generate each pattern cell")
    ap.add_argument("--grid", type=float, default=CSV_GRID,
                    help="Time between rows in CSV (sec)")
    ap.add_argument("--time-signature", type=str, default=SIGNATURE)
    args = ap.parse_args()

    try:
        num, den = map(int, args.time_signature.split("/"))
    except ValueError:
        raise SystemExit("❌  --time-signature must be in 'num/den' format")

    cells = num
    csv_files = glob.glob(os.path.join(args.folder, "*.csv"))
    if not csv_files:
        raise SystemExit(f"❌  No CSV files found in {args.folder}")

    all_hr_vals, all_sdnn_vals, all_rmssd_vals = [], [], []
    for csv_f in csv_files:
        try:
            df = pd.read_csv(csv_f, usecols=lambda c: c.upper().startswith(("HR", "SDNN", "RMSSD")))
            hr_col = df.iloc[:, 0].to_numpy()
            all_hr_vals.append(hr_col[~np.isnan(hr_col)])

            if "SDNN" in df.columns:
                sdnn_col = df["SDNN"].to_numpy()
                all_sdnn_vals.append(sdnn_col[~np.isnan(sdnn_col)])
            else:
                print(f"⚠ {os.path.basename(csv_f)}: no SDNN column – skipped")

            if "RMSSD" in df.columns:
                rmssd_col = df["RMSSD"].to_numpy()
                all_rmssd_vals.append(rmssd_col[~np.isnan(rmssd_col)])
            else:
                print(f"⚠ {os.path.basename(csv_f)}: no RMSSD column – skipped")
        except Exception as e:
            print(f"⚠ Error reading {csv_f}: {e}")

    if not all_hr_vals:
        raise SystemExit("❌  No valid HR data found in any file")

    hr_min, hr_max         = np.min(np.concatenate(all_hr_vals)),   np.max(np.concatenate(all_hr_vals))
    sdnn_min, sdnn_max     = np.min(np.concatenate(all_sdnn_vals)), np.max(np.concatenate(all_sdnn_vals))
    rmssd_min, rmssd_max   = np.min(np.concatenate(all_rmssd_vals)),np.max(np.concatenate(all_rmssd_vals))

    rules = generate_rules(hr_min, hr_max, cells)
    print(f"\n📏 Global HR zones mapped to cells and notes:")
    for thr, cell, note in rules:
        print(f"  HR ≤ {thr:.1f} → cell {cell}, note {note}")

    for csv_f in csv_files:
        grid_sec = args.grid
        midi_out = os.path.splitext(csv_f)[0] + "_drums.mid"
        csv_to_midi(csv_f, midi_out,
                    bpm=args.tempo, ppq=args.ppq,
                    seconds_per_cell=args.seconds_per_cell,
                    grid_sec=grid_sec,
                    cells=cells, denominator=den,
                    rules=rules,
                    sdnn_min=sdnn_min, sdnn_max=sdnn_max,
                    rmssd_min=rmssd_min, rmssd_max=rmssd_max)

if __name__ == "__main__":
    main()
