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
    embedding_ids = [
        "he2" + f"w{i}d{j}" for i in range(3, 7) for j in range(1, 4)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # set up a seaborn stacked barplot
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind")
    sns.set_style("ticks")

    # loop over embedding ids, find the run with the best kta and plot a bar with the initial kta for that run and the best kta
    for i, embedding_id in enumerate(embedding_ids):
        df = pd.read_csv(
            RESULTS_DIR / f"{python_results_file_name}_{embedding_id}.csv")
        df.columns = ["run_id", "epoch", "kta"]

        # find the run with the best kta
        best_run_id = df[df["epoch"] == -1]["run_id"].values[0]
        best_kta = df[df["epoch"] == -1]["kta"].values[0]
        initial_kta = df[df["run_id"] == best_run_id]["kta"].values[0]

        # plot a bar with the initial kta and the best kta
        sns.barplot(
            x=[f"{embedding_id}"],
            y=[best_kta],
            color="C1",
            ax=ax
        )
        sns.barplot(
            x=[f"{embedding_id}"],
            y=[initial_kta],
            color="C0",
            ax=ax
        )

        # annotate the bars with the best kta and the initial kta
        sns.set_context("paper", font_scale=1.1)  # Adjust the font scale

        # ...
        ax.annotate(f"{best_kta:.3f}",
                    xy=(i, best_kta),
                    # Adjust the distance from the top of the bars
                    xytext=(i, best_kta + 0.005),
                    ha="center",
                    va="bottom",
                    color="black")
        ax.annotate(f"{initial_kta:.3f}",
                    xy=(i, initial_kta),
                    # Adjust the distance from the top of the bars
                    xytext=(i, initial_kta - 0.02),
                    ha="center",
                    va="bottom",
                    color="white")

    # remove y ticks and outer frame
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)

    # add a legend with matching colors for the bars
    leg = plt.legend(["initial KTA", "best KTA"], loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, 1.1), frameon=False,
                     fontsize=12)
    leg.legend_handles[0].set_color("C0")
    leg.legend_handles[1].set_color("C1")

    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
