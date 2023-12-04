import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_directories import RESULTS_DIR, GRAPHICS_DIR

if __name__ == "__main__":
    start = time.time()

    # Load results into DataFrame
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    # [3:] removes the heading 'pp_'
    python_results_file_name = python_file_name_no_ext[3:]

    df = pd.read_csv(RESULTS_DIR / f"{python_results_file_name}.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(df["train_size"], df["mean_test_accuracy"],
                yerr=df["std_test_accuracy"], fmt='o')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test accuracy')

    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
