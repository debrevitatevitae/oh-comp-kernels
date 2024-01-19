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


jax.config.update('jax_enable_x64', False)


if __name__ == '__main__':
    start = time.time()

    np.random.seed(42)
    key = jax.random.PRNGKey(42)

    # data loading, splitting, and scaling
    df_data = pd.read_csv(PROC_DATA_DIR / 'data_labeled.csv')
    df_train, _ = train_test_split(df_data, train_size=0.3)
    X_train = df_train[['eps11', 'eps22', 'eps12']].to_numpy()
    y_train = df_train['failed'].to_numpy(dtype=np.int32)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)

    # Define the parameter grid for GridSearchCV
    cv_param_grid = {
        "C": np.logspace(-1, 5, 7)
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

            @jax.jit
            @qml.qnode(device=dev, interface="jax")
            def kernel(x1, x2):
                """x1, x2 must be JAX arrays"""
                qml.IQPEmbedding(x1, wires, n_repeats=num_layers, pattern=None)
                qml.adjoint(qml.IQPEmbedding)(
                    x2, wires, n_repeats=num_layers, pattern=None)
                return qml.expval(qml.Projector([0]*num_qubits, wires=wires))

            # create SVC and limit the number of iterations to a reasonable number
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
                f"_iqpw{num_qubits}d{num_layers}"
            df_results.to_csv(
                RESULTS_DIR / f'{results_file_name}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
