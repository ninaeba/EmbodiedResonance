#!/usr/bin/env python3
# coding: utf-8
"""
ecg_hrv_viewer.py  –  HR / HRV (ковзне 60 c, крок 0.5 c) + інтерактивний Plotly
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.signal as sg
import plotly.graph_objs as go
import neurokit2 as nk

# ─── ПАРАМЕТРИ ────────────────────────────────────────────────────
FS          = 500          # Гц (частота дискретизації ЕКГ)
WIN_SEC     = 180.0        # довжина ковзного вікна, с
STEP_SEC    = 6            # крок вікна, с
RESAMPLE_RR = 10            # Гц, для PSD
# ─────────────────────────────────────────────────────────────────


def read_ecg(path):
    """Зчитує одноколонковий CSV сигнал ЕКГ у numpy-масив."""
    return pd.read_csv(path, header=None).squeeze().astype(float).values


def bandpass_ecg(sig):
    """Бандвідт-фільтр 0.5–40 Гц для сирого ЕКГ."""
    b, a = sg.butter(4, [0.5, 40], "bandpass", fs=FS)
    return sg.filtfilt(b, a, sig)


def detect_r_peaks(sig):
    """Обробка через NeuroKit2 → масив індексів R-піків та часів (с)."""
    signals, info = nk.ecg_process(sig, sampling_rate=FS)
    r_peaks = info["ECG_R_Peaks"].astype(int)
    r_times = r_peaks / FS
    return r_peaks, r_times


def hrv_metrics(rr, r_times_window):
    """
    Розрахунок HRV-метрик:
      rr               – масив RR-інтервалів (с)
      r_times_window   – масив часів R-піків, що входять у вікно (с)
    Повертає: HR, SDNN, RMSSD, VLF, LF, HF, LF_HF, TP
    """
    # якщо мало даних — повернути NaN
    if rr.size < 2:
        return [np.nan]*8

    # 1) Часові метрики
    hr    = 60.0 / np.mean(rr)                       # bpm
    sdnn  = np.std(rr, ddof=1) * 1e3                 # мс
    rmssd = np.sqrt(np.mean(np.diff(rr)**2)) * 1e3   # мс

    # 2) Підготовка тимчасової осі для PSD
    #    t_rr співпадає з часами другого і далі піків
    t_rr = r_times_window[1:]

    # 3) Інтерполяція RR на рівномірну сітку
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1/RESAMPLE_RR)
    rri_uniform = np.interp(t_uniform, t_rr, rr)

    # 4) Обчислення PSD методом Welch
    nperseg = min(256, len(rri_uniform))
    f, pxx = sg.welch(rri_uniform,
                      fs=RESAMPLE_RR,
                      window="hann",
                      nperseg=nperseg,
                      detrend="constant")

    # 5) Функція для інтеграції з включенням верхнього краю
    def band_power(low, high):
        mask = (f >= low) & (f <= high)
        return np.trapz(pxx[mask], f[mask])

    # 6) Розрахунок абсолютних потужностей
    vlf = band_power(0.003, 0.040)
    lf  = band_power(0.040, 0.150)
    hf  = band_power(0.150, 0.400)
    tp  = band_power(0.003, 0.400)
    lf_hf = lf / hf if hf > 0 else np.nan

    return hr, sdnn, rmssd, vlf, lf, hf, lf_hf, tp


def compute_series(r_times):
    """
    Для заданого масиву часів R-піків будує DataFrame
    зі скользящими HRV-метриками.
    """
    cols = ["time", "HR", "SDNN", "RMSSD", "VLF", "LF", "HF", "TP", "LF_HF"]
    out = {c: [] for c in cols}

    start = r_times[0]
    end   = r_times[-1]

    while start + WIN_SEC <= end:
        window_mask = (r_times >= start) & (r_times <= start + WIN_SEC)
        idx = np.where(window_mask)[0]

        # беремо тільки ті RR, що віконі
        rr = np.diff(r_times[idx])
        r_times_win = r_times[idx]

        # метрики
        hr, sdnn, rmssd, vlf, lf, hf, lf_hf, tp = hrv_metrics(rr, r_times_win)
        center = start + WIN_SEC/2

        # складаємо результати
        for k, v in zip(cols,
                        [center, hr, sdnn, rmssd, vlf, lf, hf, tp, lf_hf]):
            out[k].append(v)

        start += STEP_SEC

    return pd.DataFrame(out)


def build_plot(ecg, df, title):
    """Побудова інтерактивного графіку ECG + HRV-метрики."""
    t_ecg = np.arange(len(ecg)) / FS
    fig = go.Figure()

    # сирий ЕКГ
    fig.add_trace(go.Scatter(x=t_ecg, y=ecg, name="ECG",
                             yaxis="y1", line=dict(color="steelblue")))

    # HRV-метрики
    colours = dict(HR="crimson", SDNN="darkgreen", RMSSD="orange",
                   VLF="royalblue", LF="mediumseagreen",
                   HF="goldenrod", TP="purple", LF_HF="black")
    for col in ["HR", "SDNN", "RMSSD", "VLF", "LF", "HF", "TP", "LF_HF"]:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[col], name=col, yaxis="y2",
            line=dict(color=colours[col])))

    # налаштування layout
    fig.update_layout(
        title=title,
        height=720,

        xaxis=dict(
            title="Time, s",
            rangeslider=dict(visible=True),
            range=[0, WIN_SEC]
        ),

        yaxis=dict(title="ECG (a.u.)"),

        yaxis2=dict(
            title="HR / HRV",
            overlaying="y",
            side="right",
            showgrid=False
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5
        )
    )

    fig.show()


def main(csv_path):
    ecg = read_ecg(csv_path)
    ecg_filt = bandpass_ecg(ecg)
    _, r_times = detect_r_peaks(ecg_filt)

    df_metrics = compute_series(r_times)
    build_plot(ecg, df_metrics, title=os.path.basename(csv_path))

    out_csv = csv_path.replace(".csv", "_hrv_60s.csv")
    df_metrics.to_csv(out_csv, index=False)
    print("✅ HRV table saved:", out_csv)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ecg_hrv_viewer.py <ecg_file.csv>")
        sys.exit(1)
    main(sys.argv[1])
