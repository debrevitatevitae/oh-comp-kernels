from functools import partial
import os
import pickle
import time

import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from project_directories import PICKLE_DATA_DIR, RESULTS_DIR

from utils import load_split_data

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
    inc = 1
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))
    # encode data one last time to avoid cancellations in the kernel circuit
        i = len(params) * len(wires)
        for wire in wires:
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc


if __name__ == '__main__':
    start = time.time()

    np.random.seed(42)
    key = jax.random.PRNGKey(42)

    # data loading, splitting, and scaling
    X_train, X_test, y_train, y_test = load_split_data(test_size=0.2)
    num_samples = len(y_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # convert the data into jax.numpy array
    X_train_scaled = jnp.array(X_train_scaled)
    y_train = jnp.array(y_train)
    X_test_scaled = jnp.array(X_test_scaled)
    y_test = jnp.array(y_test)

    # define the embedding kernel (with JAX)
    num_qubits = 3
    num_layers = 1
    wires = list(range(num_qubits))
    dev = qml.device('default.qubit.jax', wires=num_qubits)

    # random params or load them from pkl file
    load_params = False

    if load_params:
        with open(PICKLE_DATA_DIR / f"q_kern_kta_opt_0{num_qubits}0{num_layers}.pkl", 'rb') as params_file:
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
    cv_param_grid = {'C': np.logspace(-1, 2, 4)}

    # create SVC
    svc = SVC(kernel=partial(qml.kernels.kernel_matrix, kernel=kernel))

    # create a GridSearchCV and fit to the data
    grid_search = GridSearchCV(
        estimator=svc, param_grid=cv_param_grid, scoring='accuracy', cv=10, n_jobs=-1, refit=False, verbose=3)

    grid_search.fit(X_train_scaled, y_train)

    # store CV results in a DataFrame
    df_results = pd.DataFrame(grid_search.cv_results_)

    # Extract the mean and standaed deviation of the validation error and save to csv file
    selected_columns = ['param_C',
                        'mean_test_score', 'std_test_score']
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    file_name_width_depth = python_file_name_no_ext + \
        f"_0{num_qubits}0{num_layers}"
    results_file_name = file_name_width_depth + \
        "_trained" if load_params else file_name_width_depth + "_random"
    df_results[selected_columns].to_csv(
        RESULTS_DIR / f'{results_file_name}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
