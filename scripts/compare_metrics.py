"""Compare different accuracy metrics (Acc, J, Pr, Re, Spec) between RBF
and the trained HE2W6D3 quantum kernel. Optionally saves the results to file."""

import os
import pickle
import time
from collections import namedtuple
from functools import partial

import jax
import numpy as np
import pandas as pd
import pennylane as qml
from jax import numpy as jnp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from ohqk.project_directories import PICKLE_DIR, RESULTS_DIR
from ohqk.quantum_embeddings import trainable_embedding
from ohqk.utils import get_classification_metrics, load_split_scale_data

jax.config.update("jax_enable_x64", False)

if __name__ == "__main__":

    start = time.time()

    np.random.seed(42)
    jax_key = jax.random.PRNGKey(42)

    # run settings
    save = True
    n_reps = 5

    # data loading, splitting, and scaling. Note the 2 different scalings
    X_train, X_test, y_train, y_test = load_split_scale_data(
        test_size=0.2, scale=None, to_jax=True
    )

    N = len(X_train)

    # output dataframe
    columns = [
        "kernel",
        "batch_size",
        "accuracy",
        "jac_index",
        "precision",
        "recall",
        "spec",
    ]
    out_df = pd.DataFrame(columns=columns)

    # define rbf kernel function
    gamma_opt = 1.065

    @jax.jit
    def rbf_kernel(x1, x2):
        return jnp.exp(-gamma_opt * jnp.linalg.norm(x1 - x2, ord=2))

    # define quantum kernel function
    with open(PICKLE_DIR / "q_kern_kta_opt_he2w6d3.pkl", "rb") as f:
        params = pickle.load(f)

    embedding = partial(trainable_embedding, params=params)
    wires = list(range(6))
    dev = qml.device("default.qubit.jax", wires=wires)

    @jax.jit
    @qml.qnode(device=dev, interface="jax")
    def quantum_kernel(x1, x2):
        """x1, x2 must be JAX arrays"""
        embedding(x1, wires=wires)
        qml.adjoint(embedding)(x2, wires=wires)
        return qml.expval(qml.Projector([0] * len(wires), wires=wires))

    # add configuration for the two experiments
    experiment = namedtuple("Experiment", ["kernel", "C", "scaler"])
    exp1 = experiment(rbf_kernel, 1e7, StandardScaler())
    exp2 = experiment(
        quantum_kernel, 1e4, MinMaxScaler(feature_range=(0, np.pi))
    )

    # start experiments
    for exp in exp1, exp2:

        for batch_size in [int(N * frac) for frac in np.arange(0.1, 1.1, 0.1)]:
            print(exp.kernel.__name__, f" batch_size={batch_size}")
            accuracies = []
            jaccards = []
            precisions = []
            recalls = []
            specificities = []

            for _ in range(n_reps):
                clf = SVC(
                    kernel=partial(
                        qml.kernels.kernel_matrix, kernel=exp.kernel
                    ),
                    C=exp.C,
                    max_iter=200_000,
                    verbose=True,
                )

                idxs_selection = np.random.choice(N, size=batch_size)

                X_train_scaled_selection = exp.scaler.fit_transform(
                    X_train[idxs_selection]
                )
                y_train_selection = y_train[idxs_selection]

                clf.fit(X_train_scaled_selection, y_train_selection)

                X_test_scaled = exp.scaler.transform(X_test)

                acc, jac, prec, rec, spec = get_classification_metrics(
                    clf, X_test_scaled, y_test
                )

                accuracies.append(acc)
                jaccards.append(jac)
                precisions.append(prec)
                recalls.append(rec)
                specificities.append(spec)

            mean_acc = np.mean(accuracies)
            mean_jac = np.mean(jaccards)
            mean_prec = np.mean(precisions)
            mean_rec = np.mean(recalls)
            mean_spec = np.mean(specificities)

            out_df.loc[len(out_df)] = {
                "kernel": exp.kernel.__name__,
                "batch_size": batch_size,
                "accuracy": mean_acc,
                "jac_index": mean_jac,
                "precision": mean_prec,
                "recall": mean_rec,
                "spec": mean_spec,
            }

            print(
                f"""Metrics:
                    accuracy: {mean_acc}
                    jaccard index: {mean_jac}
                    precision: {mean_prec}
                    recall: {mean_rec}
                    specificity: {mean_spec}
                    """
            )

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    if save:
        out_df.to_csv(
            RESULTS_DIR / f"{python_file_name_no_ext}.csv", index=False
        )

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
