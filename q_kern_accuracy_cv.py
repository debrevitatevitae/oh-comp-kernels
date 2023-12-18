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
from project_directories import PICKLE_DATA_DIR, PROC_DATA_DIR, RESULTS_DIR


jax.config.update('jax_enable_x64', False)


def layer(x, params, wires, i0=0, inc=1):
    """Taken from https://github.com/thubregtsen/qhack"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring",
                  wires=wires, parameters=params[1])


def embedding(x, params, wires):
    """Adapted from https://github.com/thubregtsen/qhack"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


if __name__ == '__main__':
    start = time.time()

    np.random.seed(42)
    key = jax.random.PRNGKey(42)

    # data loading, splitting, and scaling
    df_data = pd.read_csv(PROC_DATA_DIR / 'data_labeled.csv')
    df_train, _ = train_test_split(df_data, train_size=0.1)
    X_train = df_train[['eps11', 'eps22', 'eps12']].to_numpy()
    y_train = df_train['failed'].to_numpy(dtype=np.int32)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)

    # convert the data into jax.numpy array
    X_train_scaled = jnp.array(X_train_scaled)
    y_train = jnp.array(y_train)

    # define the embedding kernel (with JAX)
    num_qubits = 6
    num_layers = 4
    wires = list(range(num_qubits))
    dev = qml.device('default.qubit.jax', wires=num_qubits)

    # random params or load them from pkl file
    load_params = True

    if load_params:
        with open(PICKLE_DATA_DIR / f"q_kern_kta_opt_he2w{num_qubits}d{num_layers}.pkl", 'rb') as params_file:
            params = pickle.load(params_file)
    else:
        _, key = jax.random.split(key)
        params = jax.random.uniform(
            key, minval=0., maxval=2*jnp.pi, shape=(num_layers, 2, num_qubits))

    @jax.jit
    @qml.qnode(device=dev, interface="jax")
    def kernel(x1, x2):
        """x1, x2 must be JAX arrays"""
        embedding(x1, params, wires)
        qml.adjoint(embedding)(x2, params, wires)
        return qml.expval(qml.Projector([0]*num_qubits, wires=wires))

    # Define the parameter grid for GridSearchCV
    cv_param_grid = {
        "C": np.logspace(-1, 2, 4)
    }

    # create SVC
    svc = SVC(kernel=partial(qml.kernels.kernel_matrix, kernel=kernel))

    # create a GridSearchCV and fit to the data
    grid_search = GridSearchCV(svc, cv_param_grid, cv=5, n_jobs=-1, verbose=3)

    grid_search.fit(X_train_scaled, y_train)

    df_results = pd.DataFrame(grid_search.cv_results_)

    # Extract the mean and standaed deviation of the validation error and save to csv file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    file_name_width_depth = python_file_name_no_ext + \
        f"_he2w{num_qubits}d{num_layers}"
    results_file_name = file_name_width_depth + \
        "_trained" if load_params else file_name_width_depth + "_random"
    df_results.to_csv(
        RESULTS_DIR / f'{results_file_name}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
