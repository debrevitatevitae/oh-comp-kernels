import argparse
from functools import partial
import os
import pickle
import time

import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from ohqk.project_directories import PICKLE_DIR, PROC_DATA_DIR, RESULTS_DIR
from ohqk.quantum_embeddings import trainable_embedding


jax.config.update('jax_enable_x64', False)


if __name__ == '__main__':
    start = time.time()

    np.random.seed(42)
    key = jax.random.PRNGKey(42)

    # Take the embedding type and the trained flag as command line arguments. The embedding type can be either 'he2' or 'iqp'.
    parser = argparse.ArgumentParser(
        description='Embedding type and trained flag')
    parser.add_argument('--embedding_type', type=str,
                        required=True, help='The type of the embedding')
    parser.add_argument('--trained', action='store_true',
                        help='Whether the embedding has trained parameters or not')
    args = parser.parse_args()

    # Move the arguments to variables (for convenience only)
    trained = args.trained
    embedding_type = args.embedding_type
    print(f"embedding_type: {embedding_type}, trained: {trained}")

    # data loading, splitting, and scaling
    df_data = pd.read_csv(PROC_DATA_DIR / 'data_labeled.csv')
    df_train, _ = train_test_split(df_data, train_size=0.3)
    X_train = df_train[['eps11', 'eps22', 'eps12']].to_numpy()
    y_train = df_train['failed'].to_numpy(dtype=np.int32)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)

    # Define the parameter grid for GridSearchCV
    cv_param_grid = {
        "C": np.logspace(-1, 4, 6)
    }

    for num_qubits in range(3, 7):
        for num_layers in range(1, 4):
            # Match X_train_scaled.shape[1] to num_qubits by cycling the columns of X_train_scaled
            X_train_scaled = np.hstack(
                [X_train_scaled[:, i % X_train_scaled.shape[1]].reshape(-1, 1) for i in range(num_qubits)])

            # print the head of X_train_scaled
            # print(X_train_scaled[:5])

            # convert the data into jax.numpy array
            X_train_scaled = jnp.array(X_train_scaled)
            y_train = jnp.array(y_train)

            # define the embedding kernel (with JAX)
            wires = list(range(num_qubits))
            dev = qml.device('default.qubit.jax', wires=num_qubits)

            embedding = None

            # define the embedding and possible parameters based on the embedding_type and trained flag
            match embedding_type, trained:
                case "he2", False:
                    # generate random parameters for HEA embedding
                    _, key = jax.random.split(key)
                    params = jax.random.uniform(
                        key, minval=0., maxval=2*jnp.pi, shape=(num_layers, 2, num_qubits))

                    embedding = partial(trainable_embedding, params=params)
                case "he2", True:
                    # load trained parameters for HEA embedding
                    pickle_file = PICKLE_DIR / \
                        f"q_kern_kta_opt_he2w{num_qubits}d{num_layers}.pkl"
                    with open(pickle_file, 'rb') as f:
                        params = pickle.load(f)
                    embedding = partial(trainable_embedding, params=params)
                case "iqp":
                    # use IQP embedding
                    embedding = partial(
                        qml.IQPEmbedding, n_repeats=num_layers, pattern=None)
                case _:
                    # return an error for invalid embedding_type and trained flag values
                    error_msg = f"Invalid embedding_type: {embedding_type} (possible values are 'he2' and 'iqp') or"
                    error_msg += f"\ninvalid trained flag: {trained}(possible values are True and False)."
                    raise ValueError(error_msg)

            @jax.jit
            @qml.qnode(device=dev, interface="jax")
            def kernel(x1, x2):
                """x1, x2 must be JAX arrays"""
                embedding(x1, wires=wires)
                qml.adjoint(embedding)(x2, wires=wires)
                return qml.expval(qml.Projector([0]*num_qubits, wires=wires))

            # create SVC
            svc = SVC(kernel=partial(qml.kernels.kernel_matrix,
                      kernel=kernel), cache_size=1000)

            # create a GridSearchCV and fit to the data and limit the time for each fit to 10 minutes
            grid_search = GridSearchCV(
                svc, cv_param_grid, cv=5, n_jobs=4, verbose=3)

            grid_search.fit(X_train_scaled, y_train)

            df_results = pd.DataFrame(grid_search.cv_results_)

            # Extract the mean and standaed deviation of the validation error and save to csv file
            python_file_name = os.path.basename(__file__)
            python_file_name_no_ext = os.path.splitext(python_file_name)[0]
            results_file_name = python_file_name_no_ext + \
                f"_{embedding_type}w{num_qubits}d{num_layers}_trained_{trained}"
            df_results.to_csv(
                RESULTS_DIR / f'{results_file_name}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
