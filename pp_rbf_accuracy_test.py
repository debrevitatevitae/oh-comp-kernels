import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ohqk.project_directories import RESULTS_DIR, GRAPHICS_DIR

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

    # Include an inset with the ratio of the number of support vectors to the training set size
    ax2 = fig.add_axes([0.5, 0.2, 0.3, 0.3])  # Adjust the coordinates here
    ax2.bar(df["train_size"], df["n_support_vectors"] /
            df["train_size"], width=40)

    # Format formulas as LaTeX formulas
    plt.rcParams['text.usetex'] = True
    ax2.set_xlabel(r'$N_{train}$')
    ax2.set_ylabel(r'$N_{SV}/N_{train}$')

    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
