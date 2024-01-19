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

    # Create a pivot tables for better visualization of the max and the variance
    pivot_table_max = df.pivot(
        index='width', columns='depth', values='max_kta')
    pivot_table_var = df.pivot(
        index='width', columns='depth', values='var_kta')

    # Create a heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(pivot_table_max, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'format': '%.3f'},
                ax=axes[0])
    axes[0].set_title("Max KTA")
    sns.heatmap(pivot_table_var, annot=True, fmt=".2e", cmap="cividis", cbar_kws={'format': '%.2e'},
                ax=axes[1])
    axes[1].set_title("Var KTA")
    # plt.title('Grid Search Results: Mean Test Score for C and gamma')
    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
