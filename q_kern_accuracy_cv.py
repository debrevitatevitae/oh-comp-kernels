from functools import partial
import os
import pickle
import time

import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from project_directories import PICKLE_DATA_DIR, RESULTS_DIR

from utils import load_split_data

jax.config.update('jax_enable_x64', False)


class GridSearchCVCustom():
    def __init__(self, estimator, param_grid, batch_size=10, cv=5, n_jobs=1) -> None:
        self.estimator = estimator
        self.param_grid = param_grid  # assumed to be a 1d list of `C` values
        self.batch_size = batch_size
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):

        def batch_iterator(array1, array2):
            # Ensure array1 and array2 have the same length
            assert len(array1) == len(array2)

            for start in range(0, len(array1), self.batch_size):
                end = start + self.batch_size
                yield array1[start:end], array2[start:end]

        self.cv_results_ = {
            "param_C": [],
            "mean_test_score": [],
            "std_test_score": []
        }

        for param_C in self.param_grid:
            print(f"C={param_C:e}")
            test_scores = np.array([])
            estimator = self.estimator.set_params(C=param_C)
            for X_batch, y_batch in batch_iterator(X, y):
                # exclude very small batches
                if len(y_batch) > self.cv:
                    batch_cv_scores = cross_val_score(
                        estimator, X_batch, y_batch, cv=self.cv, n_jobs=self.n_jobs, verbose=3)
                    test_scores = np.append(test_scores, batch_cv_scores)

            self.cv_results_["param_C"].append(param_C)
            self.cv_results_["mean_test_score"].append(np.mean(test_scores))
            self.cv_results_["std_test_score"].append(np.std(test_scores))


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
    num_layers = 2
    wires = list(range(num_qubits))
    dev = qml.device('default.qubit.jax', wires=num_qubits)

    # random params or load them from pkl file
    load_params = False

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
