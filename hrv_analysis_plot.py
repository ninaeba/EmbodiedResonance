"""
USAGE
-----
$ python hrv_analysis_new.py ecg.csv              # default, with band‑pass
$ python hrv_analysis_new.py ecg.csv --nofilt     # **raw ECG, no band‑pass**

The flag `--nofilt` (or `-n`) disables the Butterworth band‑pass stage that
normally cleans the raw ECG.  Everything else (R‑peak detection, HRV metrics,
plots) stays identical.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.signal as sg
import neurokit2 as nk

# ----------------------------------------------------------------------
FS_ECG   = 500                          # Hz, raw ECG
BP_ECG   = (0.5, 40)                    # band‑pass (Butter 4‑th)

TIME_WIN, TIME_STEP = 60.0, 1.0       # s  (HR, SDNN, RMSSD)
FREQ_WIN, FREQ_STEP = 300.0, 30.0       # s  (VLF/LF/HF)
FGRID = np.arange(0.003, 0.401, 0.001)  # Hz

BANDS = dict(VLF=(.003, .04), LF=(.04, .15), HF=(.15, .40))

# ------------------------------ ECG -----------------------------------
class ECG:
    """ECG helper: load, *optionally* band‑pass, detect R‑peaks."""

    def __init__(self, fs=FS_ECG, bp=BP_ECG, use_filter: bool = True):
        self.fs = fs
        self.use_filter = use_filter
        if use_filter:
            self.b, self.a = sg.butter(4, np.array(bp)/(fs/2), 'bandpass')

    def load(self, fname: Path) -> np.ndarray:
        sig = pd.read_csv(fname, header=None).squeeze().astype(float).values
        if sig.size < 10 * self.fs:
            raise ValueError("Signal shorter than 10 s.")
        return sig

    def filt(self, sig):
        if not self.use_filter:
            return sig
        return sg.filtfilt(self.b, self.a, sig)

    def r_peaks(self, sig_f):
        _, info = nk.ecg_process(sig_f, sampling_rate=self.fs)
        idx = info["ECG_R_Peaks"].astype(int)
        return idx, idx / self.fs                     # indices, times [s]


# ------------------------------ HRV ------------------------------------
class HRV:
    def __init__(self, tw=TIME_WIN, ts=TIME_STEP,
                 fw=FREQ_WIN, fstep=FREQ_STEP, fgrid=FGRID):
        self.tw, self.ts = tw, ts
        self.fw, self.fstep = fw, fstep
        self.fgrid = fgrid

    # ---- helpers ----
    @staticmethod
    def time_metrics(rr):
        if rr.size < 2: return np.nan, np.nan, np.nan
        hr    = 60 / rr.mean()
        sdnn  = rr.std(ddof=1)          * 1000
        rmssd = np.sqrt(np.mean(np.diff(rr)**2)) * 1000
        return hr, sdnn, rmssd

    def lomb_bandpowers(self, rr, t_rr):
        if rr.size < 3:
            return np.nan, np.nan, np.nan, np.nan          # VLF LF HF LF/HF

        # Lomb–Scargle (без normalize!)
        pxx = sg.lombscargle(t_rr, rr - rr.mean(),
                             2*np.pi*self.fgrid, normalize=False) / (2*np.pi)

        df = self.fgrid[1]-self.fgrid[0]
        pwr = {band: pxx[(self.fgrid>=lo)&(self.fgrid<=hi)].sum()*df
               for band,(lo,hi) in BANDS.items()}

        lf_hf = pwr['LF']/pwr['HF'] if pwr['HF']>1e-12 else np.nan
        return pwr['VLF'], pwr['LF'], pwr['HF'], lf_hf

    # ---- builders ----
    def time_series(self, r_t):
        starts = np.arange(r_t[0], r_t[-1]-self.tw+self.ts, self.ts)
        rows   = []
        for st in starts:
            sel = (r_t >= st) & (r_t <= st+self.tw)
            if sel.sum() < 2: continue
            rr = np.diff(r_t[sel])
            hr, sdnn, rmssd = self.time_metrics(rr)
            rows.append([st+self.tw/2, hr, sdnn, rmssd])
        return pd.DataFrame(rows, columns=['t','HR','SDNN','RMSSD'])

    def freq_series(self, r_t):
        starts = np.arange(r_t[0], r_t[-1]-self.fw+self.fstep, self.fstep)
        rows   = []
        for st in starts:
            sel = (r_t >= st) & (r_t <= st+self.fw)
            if sel.sum() < 3: continue
            rr   = np.diff(r_t[sel])
            t_rr = (r_t[sel] - r_t[sel][0])[1:]
            vlf, lf, hf, lf_hf = self.lomb_bandpowers(rr, t_rr)
            rows.append([st+self.fw/2, vlf, lf, hf, lf_hf])
        return pd.DataFrame(rows, columns=['t','VLF','LF','HF','LF_HF'])

    def compute(self, r_t):
        tdf, fdf = self.time_series(r_t), self.freq_series(r_t)
        if tdf.empty and fdf.empty:
            return pd.DataFrame()

        # align on 1‑second grid
        grid_start = max(tdf.t.min(), fdf.t.min())
        grid_end   = min(tdf.t.max(), fdf.t.max())
        grid = np.arange(grid_start, grid_end+1, 1.0)

        out = pd.DataFrame({'t': grid})
        for col, src in [('HR',tdf), ('SDNN',tdf), ('RMSSD',tdf),
                         ('VLF',fdf), ('LF',fdf), ('HF',fdf), ('LF_HF',fdf)]:
            if col in src.columns and not src.empty:
                out[col] = np.interp(grid, src.t, src[col])
        return out


# ------------------------------ PLOT -----------------------------------
COLORS = dict(HR='#d62728', SDNN='#2ca02c', RMSSD='#ff7f0e',
              VLF='#1f77b4', LF='#17becf', HF='#bcbd22', LF_HF='#7f7f7f')

def plot(ecg_f, hrv_df, fs=FS_ECG, title="HRV (Lomb)"):
    t = np.arange(ecg_f.size)/fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ecg_f, name='ECG', yaxis='y1',
                             line=dict(color='steelblue')))

    for col in hrv_df.columns.drop('t'):
        fig.add_trace(go.Scatter(x=hrv_df.t, y=hrv_df[col], name=col,
                                 yaxis='y2', line=dict(color=COLORS.get(col,'grey'))))

    fig.update_layout(title=title, height=750, hovermode='x unified',
        xaxis=dict(title='Time (s)', rangeslider=dict(visible=True, thickness=.05)),
        yaxis=dict(title='ECG', side='left'),
        yaxis2=dict(title='HRV', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', x=.5, y=1.04, xanchor='center'))
    fig.show()


# ------------------------------ MAIN -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="ECG → HRV (Lomb) pipeline")
    parser.add_argument('csv', help='ECG CSV (single column, raw signal)')
    parser.add_argument('-n','--nofilt', action='store_true',
                        help='**Disable band‑pass filter** (process raw ECG)')
    args = parser.parse_args()

    path = Path(args.csv)
    ecg = ECG(use_filter=not args.nofilt)

    sig  = ecg.load(path)
    sigF = ecg.filt(sig)

    peaks, r_times = ecg.r_peaks(sigF)
    print(f"Detected {peaks.size} R‑peaks.  Filter: {'OFF' if args.nofilt else 'ON'}")

    hrv_df = HRV().compute(r_times)
    print(f"Computed HRV rows: {len(hrv_df)}")

    out = path.with_suffix('').with_suffix('.hrv_lomb.csv')
    hrv_df.to_csv(out, index=False, float_format='%.6f')
    print(f"Saved → {out.name}")

    plot(sigF, hrv_df, title=f"{path.stem} – HRV (Lomb){' – RAW' if args.nofilt else ''}")


if __name__ == '__main__':
    main()
