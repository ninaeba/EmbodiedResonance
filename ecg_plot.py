import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import sys

def plot_scrollable_ecg(file_path, sampling_rate=500, window_seconds=10):
    data = pd.read_csv(file_path, header=None).squeeze()
    total_samples = len(data)
    window_size = sampling_rate * window_seconds
    time = np.arange(total_samples) / sampling_rate
    global_min = np.min(data)
    global_max = np.max(data)
    global_mean = np.mean(data)
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    l, = ax.plot(np.arange(window_size), data[:window_size], color='blue')
    ax.set_xlabel("Sample number")
    ax.set_ylabel("ECG Value")
    ax.set_title(f"ECG View: {os.path.basename(file_path)}")
    time_text = ax.text(0.01, 0.95, '', transform=ax.transAxes, fontsize=10, va='top')
    stats_text = ax.text(0.99, 0.95, '', transform=ax.transAxes, fontsize=10, va='top', ha='right')
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Scroll', 0, total_samples - window_size, valinit=0, valstep=1)

    def update(val):
        start = int(slider.val)
        end = start + window_size
        x_vals = np.arange(start, end)
        y_vals = data[start:end]

        l.set_xdata(x_vals)
        l.set_ydata(y_vals)
        ax.set_xlim(start, end)
        ax.set_ylim(np.min(y_vals), np.max(y_vals))

        time_text.set_text(f"Time: {start / sampling_rate:.2f} s")
        stats_text.set_text(
            f"Min: {global_min:.0f}  Max: {global_max:.0f}  Mean: {global_mean:.0f}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()

# Set in terminal the path to a single-lead ECG file 
def main():
    if len(sys.argv) < 2:
        print("❌ Please provide a path to an ECG CSV file.")
        return
    file_path = sys.argv[1]
    sampling_rate = 500
    window_seconds = 10
    plot_scrollable_ecg(file_path, sampling_rate, window_seconds)

if __name__ == "__main__":
    main()


