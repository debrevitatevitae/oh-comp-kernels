import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ohqk.kta_classical import KernelTargetAlignmentLoss, rbf_kernel
from ohqk.project_directories import PROC_DATA_DIR, RESULTS_DIR

if __name__ == "__main__":
    start = time.time()

    np.random.seed(42)

    save = True

    # data loading, splitting and scaling
    df_data = pd.read_csv(PROC_DATA_DIR / "data_labeled.csv")
    df_train, _ = train_test_split(df_data, train_size=0.8)
    X_train = df_train[["eps11", "eps22", "eps12"]].to_numpy()
    y_train = df_train["failed"].to_numpy(dtype=np.int32)
    N = len(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # convert to torch tensors
    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int32)

    # optimization hyperparameters
    num_epochs = 500
    batch_size = 5
    lr = 1e-2
    gamma = torch.tensor(0.1, requires_grad=True)  # initial gamma
    opt = torch.optim.Adam([gamma], lr)
    loss_function = KernelTargetAlignmentLoss(rbf_kernel)

    # how many times to compute the batched kta at checkpoint (for reporting)
    num_reps_at_checkpoint = 500
    epochs_to_checkpoint = 10
    columns = ["kta", "gamma"]
    rows = list(
        range(0, num_epochs + epochs_to_checkpoint, epochs_to_checkpoint)
    )
    df = pd.DataFrame(columns=columns, index=rows)

    # batching function
    def batch_data_and_labels():
        idxs_batch = np.random.choice(N, size=batch_size)
        return X_train_scaled[idxs_batch], y_train[idxs_batch]

    # function to compute an average kta at checkpoints
    def compute_avg_kta():
        kta_avg = 0.0
        for _ in range(num_reps_at_checkpoint):
            X_train_scaled_batch, y_train_batch = batch_data_and_labels()
            kta_avg -= (
                1
                / num_reps_at_checkpoint
                * loss_function(X_train_scaled_batch, y_train_batch, gamma)
            )
        return kta_avg

    # initial kta
    kta_avg = compute_avg_kta()
    print(f"Initial average KTA ={kta_avg:.5f} , initial gamma ={gamma:.3f}")
    df.loc[0, "kta"] = kta_avg.item()
    df.loc[0, "gamma"] = gamma.item()

    # optimization loop
    for ep in range(num_epochs):
        # randomly select batch
        X_train_scaled_batch, y_train_batch = batch_data_and_labels()

        # loss computation
        loss = loss_function(X_train_scaled_batch, y_train_batch, gamma)

        # gradient computation and optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # compute average alignment over some batches, store in DataFrame and report
        if (ep + 1) % epochs_to_checkpoint == 0:
            kta_avg = compute_avg_kta()

            df.loc[ep + 1, "kta"] = kta_avg.item()
            df.loc[ep + 1, "gamma"] = gamma.item()
            print(
                f"Epoch {ep+1}, average KTA = {kta_avg:.5f}, gamma={gamma:.3f}"
            )

    # save DataFrame to a csv file named after this Python file
    if save:
        python_file_name = os.path.basename(__file__)
        python_file_name_no_ext = os.path.splitext(python_file_name)[0]
        df.to_csv(
            RESULTS_DIR / f"{python_file_name_no_ext}.csv",
            index=True,
            index_label="epochs",
        )

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
