import os
import time

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from ohqk.project_directories import GRAPHICS_DIR, RESULTS_DIR

if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    start = time.time()

    # Load results into DataFrame
    python_results_file_name = RESULTS_DIR / "rbf_kta_opt.csv"

    df = pd.read_csv(RESULTS_DIR / python_results_file_name)

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Set the y-tick formatting function
    def format_y_ticks(value, _):
        return f"{value:.3f}"

    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

    # Plot
    ax.plot(df["epochs"], df["kta"], linewidth=1.5, color="blue")
    ax.set_xlabel("epochs")
    ax.set_ylabel("KTA")
    ax.grid()

    # Adjust layout
    fig.tight_layout()

    # Show the plot
    fig.savefig(GRAPHICS_DIR / "pp_rbf_kta_opt_paper.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
