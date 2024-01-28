import os
import pickle
import time
from functools import partial

import jax
import numpy as np
import pandas as pd
import pennylane as qml
from jax import numpy as jnp
from pennylane.templates.embeddings import IQPEmbedding
from sklearn.svm import SVC

from ohqk.project_directories import PICKLE_DIR, RESULTS_DIR
from ohqk.quantum_embeddings import trainable_embedding
from ohqk.utils import load_split_scale_data, match_shape_to_num_qubits

jax.config.update("jax_enable_x64", False)


if __name__ == "__main__":
    start = time.time()

    np.random.seed(42)
    jax_key = jax.random.PRNGKey(42)

    # data loading, splitting, and scaling
    X_train_scaled, X_test_scaled, y_train, y_test = load_split_scale_data(
        test_size=0.2, scale="angle", to_jax=True
    )

    N = len(X_train_scaled)

    df_best_classfiers = pd.read_csv(
        RESULTS_DIR / "best_scoring_embeddings.csv"
    )

    # create pandas DataFrame to store results
    columns = [
        "embedding",
        "train_size",
        "mean_test_accuracy",
        "std_test_accuracy",
    ]
    out_df = pd.DataFrame(columns=columns)

    for _, row in df_best_classfiers.iterrows():
        print(f"\nEmbedding: {row['embedding']}")

        # Match the shape of the input arrays to the number of qubits
        X_train_scaled = match_shape_to_num_qubits(
            X_train_scaled, row["num_qubits"]
        )
        X_test_scaled = match_shape_to_num_qubits(
            X_test_scaled, row["num_qubits"]
        )

        # define the embedding kernel (with JAX)
        wires = list(range(row["num_qubits"]))
        dev = qml.device("default.qubit.jax", wires=wires)

        match row["embedding"]:
            case "iqp":
                embedding = partial(
                    IQPEmbedding, n_repeats=row["num_layers"], pattern=None
                )
            case "he2_trained_False":
                params = jax.random.uniform(
                    jax_key,
                    minval=0.0,
                    maxval=2 * jnp.pi,
                    shape=(row["num_layers"], 2, row["num_qubits"]),
                )
                embedding = partial(trainable_embedding, params=params)
            case "he2_trained_True":
                with open(PICKLE_DIR / row["var_params_filename"], "rb") as f:
                    params = pickle.load(f)
                embedding = partial(trainable_embedding, params=params)
            case _:
                raise ValueError(
                    "Something went wrong. The embedding is invalid."
                )

        @jax.jit
        @qml.qnode(device=dev, interface="jax")
        def kernel(x1, x2):
            """x1, x2 must be JAX arrays"""
            embedding(x1, wires=wires)
            qml.adjoint(embedding)(x2, wires=wires)
            return qml.expval(
                qml.Projector([0] * row["num_qubits"], wires=wires)
            )

        # define the SVC.
        clf = SVC(
            kernel=partial(qml.kernels.kernel_matrix, kernel=kernel),
            C=row["param_C"],
        )

        # declare some training set sizes
        train_sizes = [int(N * frac) for frac in np.arange(0.1, 1.1, 0.1)]

        # for each of the train sizes, repeat a training and compute the test accuracy
        num_reps = 5
        for i, ts in enumerate(train_sizes):
            accuracies = []
            for _ in range(num_reps):
                idxs_selection = np.random.choice(N, size=ts)
                X_train_scaled_selection = X_train_scaled[idxs_selection]
                y_train_selection = y_train[idxs_selection]
                # create SVC with optimal hyperparameters
                # fit the classifier
                clf.fit(X_train_scaled_selection, y_train_selection)
                accuracies.append(clf.score(X_test_scaled, y_test))

            out_df.loc[len(out_df)] = {
                "embedding": row["embedding"],
                "train_size": ts,
                "mean_test_accuracy": np.mean(accuracies),
                "std_test_accuracy": np.std(accuracies),
            }

            # print mean and std of test accuracy
            print(
                f"Train size: {ts}, mean test accuracy: {np.mean(accuracies)}, std test accuracy: {np.std(accuracies)}"
            )

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    out_df.to_csv(RESULTS_DIR / f"{python_file_name_no_ext}.csv", index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
