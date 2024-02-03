import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ohqk.kta_classical import KernelTargetAlignmentLoss, rbf_kernel
from ohqk.project_directories import RESULTS_DIR
from ohqk.utils import load_split_data

if __name__ == "__main__":
    start = time.time()

    np.random.seed(42)

    # data loading, splitting and scaling
    X_train_scaled, _, y_train, _ = load_split_data(
        test_size=0.2, scale="standard", to_torch=True, to_jax=False
    )

    # optimization hyperparameters
    num_epochs = 100
    batch_size = 471  # 30% of the training dataset dimension
    lr = 1e-1
    gamma = torch.tensor(0.1, requires_grad=True)  # initial gamma
    opt = torch.optim.Adam([gamma], lr)
    loss_function = KernelTargetAlignmentLoss(rbf_kernel)

    # DataFrame to store target alignment and parameters at checkpoints
    num_checkpoints = 20
    # how many times to compute the batched kta at checkpoint (for reporting)
    num_reps_at_checkpoint = 10
    epochs_in_checkpoint = num_epochs // num_checkpoints
    columns = ["kta", "gamma"]
    rows = ["Initial"]
    rows.extend(
        [
            f"Epoch no {n}"
            for n in range(
                epochs_in_checkpoint,
                num_epochs + epochs_in_checkpoint,
                epochs_in_checkpoint,
            )
        ]
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
    df.loc["Initial", "kta"] = kta_avg.item()
    df.loc["Initial", "gamma"] = gamma.item()

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
        if (ep + 1) % epochs_in_checkpoint == 0:
            kta_avg = compute_avg_kta()

            df.loc[f"Epoch no {ep+1}", "kta"] = kta_avg.item()
            df.loc[f"Epoch no {ep+1}", "gamma"] = gamma.item()
            print(
                f"Epoch {ep+1}, average KTA = {kta_avg:.5f}, gamma={gamma:.3f}"
            )

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    df.to_csv(RESULTS_DIR / f"{python_file_name_no_ext}.csv", index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
