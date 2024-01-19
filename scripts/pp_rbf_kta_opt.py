import os
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

import pandas as pd

from ohqk.project_directories import RESULTS_DIR, GRAPHICS_DIR


if __name__ == '__main__':
    start = time.time()

    # Load results into DataFrame
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    # [3:] removes the heading 'pp_'
    python_results_file_name = python_file_name_no_ext[3:]

    df = pd.read_csv(RESULTS_DIR / f"{python_results_file_name}.csv")

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Select the max KTA value and the corresponding number of epochs
    kta_max = df['kta'].max()
    epochs_kta_max = df['kta'].argmax() * 5

    # Set the y-tick formatting function
    def format_y_ticks(value, _):
        return f'{value:.3f}'

    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

    # Plot
    ax.plot(range(0, 105, 5), df['kta'], linewidth=1.5, color='blue')
    ax.set_xticks([0, epochs_kta_max, 100])
    ax.set_yticks([df['kta'][0], kta_max])
    ax.set_xlabel('epochs')
    ax.set_ylabel('KTA')

    # Adjust layout
    fig.tight_layout()

    # Show the plot
    fig.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
