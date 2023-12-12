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
    embedding_ids = ["_0301", "_0302", "_0601", "_0602", "_0904"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#feebe2', '#fbb4b9', '#f768a1', '#c51b8a', '#7a0177']

    for i, embd_id in enumerate(embedding_ids):
        f = python_results_file_name + embd_id
        df = pd.read_csv(RESULTS_DIR / f"{f}.csv", dtype={
            "run_id": int,
            "epoch": int,
            "kta": float
        })

        # select best run optimization
        hist_df = df.iloc[:-1]
        best_run = df["run_id"].iloc[-1]
        only_best_run = hist_df["run_id"] == best_run
        best_run_df = hist_df[only_best_run]

        # add to plot
        ax.plot(best_run_df["epoch"], best_run_df["kta"], color=colors[i])

        # annotate label
        ax.text(best_run_df["epoch"].iloc[-1] + 10,
                best_run_df["kta"].iloc[-1], "qk" + embd_id, va="center")

        # find max and annotate initial and max
        max_kta_idx = best_run_df["kta"].argmax()
        max_kta = best_run_df["kta"].iloc[max_kta_idx]
        max_kta_epoch = best_run_df["epoch"].iloc[max_kta_idx]
        ax.annotate(f"{best_run_df['kta'].iloc[0]:.3f}", (best_run_df["epoch"].iloc[0], best_run_df["kta"].iloc[0]),
                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7)
        ax.annotate(f"{max_kta:.3f}", (max_kta_epoch, max_kta),
                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7)

    ax.set_xticks([0, 499])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("epoch")  # remember that this is an average

    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
