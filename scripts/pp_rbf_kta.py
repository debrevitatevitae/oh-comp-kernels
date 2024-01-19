import os
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

import pandas as pd

from ohqk.project_directories import RESULTS_DIR, GRAPHICS_DIR


kta_full_train_set = 0.06969  # from rbf_kta.py

if __name__ == '__main__':
    start = time.time()

    # Load results into DataFrame
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    # [3:] removes the heading 'pp_'
    python_results_file_name = python_file_name_no_ext[3:]

    df = pd.read_csv(RESULTS_DIR / f"{python_results_file_name}.csv")

    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # Flatten the 2D array of axes for easier iteration
    axes = axes.flatten()

    # Find the KTA extreme values to use as x_lims
    kta_min = df.min().min()
    kta_max = df.max().max()

    # Formatting function for the precision to show in the x-ticks
    def format_x_ticks(value, _):
        return f'{value:.2f}'

    # Plot each column
    for i, column in enumerate(df.columns):
        ax = axes[i]
        ax.hist(df[column], bins=10, color='skyblue', edgecolor='black')
        ax.axvline(x=kta_full_train_set, color='red',
                   linestyle='dashed', linewidth=2, label='full train set')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(column)
        # format every x-axis
        ax.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))

    # Decorate axes
    axes[0].set_ylabel('Frequency')
    axes[0].set_yticks(np.linspace(0, 40, 8, dtype=np.int16))
    axes[3].set_xlabel('KTA')
    axes[3].set_xticks(np.linspace(kta_min, kta_max, 6))
    axes[3].set_yticks(np.linspace(0, 40, 8, dtype=np.int16))
    axes[3].set_ylabel('Frequency')
    axes[4].set_xlabel('KTA')
    axes[4].set_xticks(np.linspace(kta_min, kta_max, 6))
    axes[5].set_xlabel('KTA')
    axes[5].set_xticks(np.linspace(kta_min, kta_max, 6))
    axes[5].legend()

    # Adjust layout to prevent overlapping titles
    fig.tight_layout()

    # Show the plot
    fig.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
