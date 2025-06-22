#!/usr/bin/env python3
"""
ECG ➜ HRV analysis (Lomb–Scargle) — per-metric windows + 1-s grid
-----------------------------------------------------------------
Run:
    python hrv_analysis_plot.py ecg.csv
    python hrv_analysis_plot.py ecg.csv -n     # raw ECG (no filter)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.signal as sg
import neurokit2 as nk

# ------------------------------------------------------------------
#                     CONSTANTS & CONFIGURATION
# ------------------------------------------------------------------
FS_ECG: int = 500                       # sampling rate [Hz]
BP_ECG: tuple[float, float] = (0.5, 40) # Butterworth band-pass

# metric-specific windows / steps (s) ------------------------------
TIME_CONF = {"HR": (6.4, 0.1), "SDNN": (120., 0.1), "RMSSD": (120, 0.1)}
FREQ_CONF = {m: (180., 0.1) for m in ["VLF", "LF", "HF", "LF_HF"]}

FGRID = np.arange(0.003, 0.401, 0.001)  # frequency grid
BANDS = {"VLF": (0.003, 0.040), "LF": (0.040, 0.150), "HF": (0.150, 0.400)}

int_p_g = 0.1

# ------------------------------------------------------------------
#                               ECG
# ------------------------------------------------------------------
class ECG:
    def __init__(self, fs: int = FS_ECG, bp: tuple[float, float] = BP_ECG,
                 use_filter: bool = True) -> None:
        self.fs, self.use_filter = fs, use_filter
        if use_filter:
            self.b, self.a = sg.butter(4, np.array(bp)/(fs/2), "bandpass")

    def load(self, fname: Path) -> np.ndarray:
        sig = pd.read_csv(fname, header=None).squeeze().astype(float).values
        if sig.size < 10*self.fs:
            raise ValueError("ECG shorter than 10 s.")
        return sig

    def filt(self, sig: np.ndarray) -> np.ndarray:
        return sg.filtfilt(self.b, self.a, sig) if self.use_filter else sig

    def r_peaks(self, sig_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, info = nk.ecg_process(sig_f, sampling_rate=self.fs)
        idx = info["ECG_R_Peaks"].astype(int)
        return idx, idx/self.fs

# ------------------------------------------------------------------
#                               HRV
# ------------------------------------------------------------------
class HRV:
    # ---------- helpers ----------
    @staticmethod
    def time_metrics(rr: np.ndarray) -> tuple[float, float, float]:
        if rr.size < 2:
            return np.nan, np.nan, np.nan
        hr = 60./rr.mean()
        sdnn = rr.std(ddof=1)*1000.
        rmssd = np.sqrt(np.mean(np.diff(rr)**2))*1000.
        return hr, sdnn, rmssd

    @staticmethod
    def lomb(rr: np.ndarray, t_rr: np.ndarray) -> tuple[float, float, float, float]:
        if rr.size < 3:
            return (np.nan,)*4
        pxx = sg.lombscargle(t_rr, rr-rr.mean(), 2*np.pi*FGRID,
                             normalize=False)/(2*np.pi)
        df = FGRID[1]-FGRID[0]
        pwr = {b: pxx[(FGRID>=lo)&(FGRID<=hi)].sum()*df
               for b,(lo,hi) in BANDS.items()}
        lf_hf = pwr["LF"]/pwr["HF"] if pwr["HF"]>1e-12 else np.nan
        return pwr["VLF"], pwr["LF"], pwr["HF"], lf_hf

    # ---------------- main API -----------------
    def compute(self, r_t: np.ndarray,
                experiment_end: float = 1440.0) -> pd.DataFrame:
        """HRV table на 1-сек. сітці до кінця експерименту (1440 с)."""
        per_metric: dict[str, pd.DataFrame] = {}

        # ---- time-domain ----
        for m, (win, step) in TIME_CONF.items():
            ends = np.arange(0.0, experiment_end + 1e-9, step)
            rows = []
            for t_end in ends:
#                win_eff = min(win, t_end)
#                t_start = t_end - win_eff

                if t_end < win:
                    continue
                t_start = t_end - win

                sel = (r_t >= t_start) & (r_t <= t_end)
                if sel.sum() < 4:
                    continue
                rr = np.diff(r_t[sel])
                hr, sdnn, rmssd = self.time_metrics(rr)
                val = {"HR": hr, "SDNN": sdnn, "RMSSD": rmssd}[m]
                rows.append([t_end, val])
            if rows:
                per_metric[m] = pd.DataFrame(rows, columns=["t", m])

        # ---- freq-domain ----
        for m, (win, step) in FREQ_CONF.items():
            ends = np.arange(0.0, experiment_end + 1e-9, step)
            rows = []
            for t_end in ends:
#                win_eff = min(win, t_end)
#                t_start = t_end - win_eff
                
                if t_end < win:
                    continue
                t_start = t_end - win

                sel = (r_t >= t_start) & (r_t <= t_end)
                if sel.sum() < 4:
                    continue
                rr  = np.diff(r_t[sel])
                t_rr = (r_t[sel] - r_t[sel][0])[1:]
                vlf, lf, hf, lf_hf = self.lomb(rr, t_rr)
                val = {"VLF": vlf, "LF": lf, "HF": hf, "LF_HF": lf_hf}[m]
                rows.append([t_end, val])
            if rows:
                per_metric[m] = pd.DataFrame(rows, columns=["t", m])

        # -------- merge + interpolation ---------
        if not per_metric:
            return pd.DataFrame()

        grid = np.arange(0.0, experiment_end + int_p_g - 1, int_p_g)   # ← 2) до 1440 с
        out = pd.DataFrame({"t": grid})

        for m, df in per_metric.items():
            out[m] = np.interp(grid, df.t, df[m], left=np.nan, right=np.nan)
        return out

# ------------------------------------------------------------------
#                             PLOTTING
# ------------------------------------------------------------------
COLORS = {"HR":"#d62728","SDNN":"#2ca02c","RMSSD":"#ff7f0e",
          "VLF":"#1f77b4","LF":"#17becf","HF":"#bcbd22","LF_HF":"#7f7f7f"}
def plot(ecg_f: np.ndarray, hrv_df: pd.DataFrame, fs: int = FS_ECG,
         title:str="HRV analysis") -> None:
    t_ecg = np.arange(ecg_f.size)/fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_ecg,y=ecg_f,name="ECG",yaxis="y1",
                             line=dict(color="steelblue")))
    for col in hrv_df.columns.drop("t"):
        fig.add_trace(go.Scatter(x=hrv_df.t,y=hrv_df[col],name=col,yaxis="y2",
                                 line=dict(color=COLORS.get(col,"grey"))))
    fig.add_vrect(x0=0,x1=360,fillcolor="rgba(173,216,230,0.25)",line_width=0,
                  layer="below",annotation_text="REST",annotation_position="top left")
    fig.add_vrect(x0=360,x1=1080,fillcolor="rgba(255,182,193,0.25)",line_width=0,
                  layer="below",annotation_text="STRESS",annotation_position="top left")
    fig.add_vrect(x0=1080,x1=1440,fillcolor="rgba(173,216,230,0.25)",line_width=0,
                  layer="below",annotation_text="RECOVER",annotation_position="top left")
    fig.update_layout(title=title,height=750,hovermode="x unified",
        xaxis=dict(title="Time (s)",rangeslider=dict(visible=True,thickness=0.05)),
        yaxis=dict(title="ECG",side="left"),
        yaxis2=dict(title="HRV",overlaying="y",side="right",showgrid=False),
        legend=dict(orientation="h",x=0.5,y=1.04,xanchor="center"))
    fig.show()

# ------------------------------------------------------------------
#                              MAIN
# ------------------------------------------------------------------
def main() -> None:
    ap=argparse.ArgumentParser(description="ECG ➜ HRV (per-metric windows+grid)")
    ap.add_argument("csv", help="ECG CSV (single column, raw signal)")
    ap.add_argument("-n","--nofilt",action="store_true",
                    help="Disable band-pass (use raw ECG)")
    args=ap.parse_args()

    path=Path(args.csv)
    ecg=ECG(use_filter=not args.nofilt)

    raw=ecg.load(path)
    filt=ecg.filt(raw)

    peaks,r_times=ecg.r_peaks(filt); r_times-=r_times[0]
    print(f"R-peaks: {peaks.size}")

    hrv=HRV(); hrv_df=hrv.compute(r_times)
    print(f"HRV rows: {len(hrv_df)}")

    out=path.with_suffix("").with_suffix(".hrv_lomb.csv")
    hrv_df.to_csv(out,index=False,float_format="%.6f")
    print(f"Saved ➜ {out.name}")

    plot(filt,hrv_df,title=f"{path.stem} – HRV {'RAW' if args.nofilt else ''}")

if __name__=="__main__":
    main()
