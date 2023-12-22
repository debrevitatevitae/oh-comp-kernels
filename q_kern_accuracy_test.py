from functools import partial
import os
import pickle
import time

import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.model_selection import train_test_split
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
    jax_key = jax.random.PRNGKey(42)

    # data loading, splitting, and scaling
    df_data = pd.read_csv(PROC_DATA_DIR / 'data_labeled.csv')
    df_train, df_test = train_test_split(df_data, test_size=0.2)

    X_train = df_train[['eps11', 'eps22', 'eps12']].to_numpy()
    y_train = df_train['failed'].to_numpy(dtype=np.int32)
    X_test = df_test[['eps11', 'eps22', 'eps12']].to_numpy()
    y_test = df_test['failed'].to_numpy(dtype=np.int32)

    N = len(y_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # convert the data into jax.numpy array
    X_train_scaled = jnp.array(X_train_scaled)
    y_train = jnp.array(y_train)
    X_test_scaled = jnp.array(X_test_scaled)
    y_test = jnp.array(y_test)

    # define the embedding kernel (with JAX)
    num_qubits = 6
    num_layers = 4
    wires = list(range(num_qubits))
    dev = qml.device('default.qubit.jax', wires=num_qubits)

    # load the optimal parameters
    with open(PICKLE_DATA_DIR / f"q_kern_kta_opt_he2w{num_qubits}d{num_layers}.pkl", 'rb') as params_file:
        params = pickle.load(params_file)

    @jax.jit
    @qml.qnode(dev, interface="jax")
    def kernel(x1, x2):
        """x1 and x2 must be JAX arrays"""
        embedding(x1, params, wires)
        qml.adjoint(embedding)(x2, params, wires)
        return qml.expval(qml.Projector([0]*num_qubits, wires=wires))

    # define the SVC, C=10.0 is the optimal hyperparameter
    clf = SVC(kernel=partial(qml.kernels.kernel_matrix, kernel=kernel), C=100.0)

    # create pandas DataFrame to store results
    columns = ["train_size", "mean_test_accuracy", "std_test_accuracy"]
    df = pd.DataFrame(columns=columns)

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

        df.loc[len(df)] = {
            "train_size": ts,
            "mean_test_accuracy": np.mean(accuracies),
            "std_test_accuracy": np.std(accuracies)
        }

        # print mean and std of test accuracy
        print(
            f"Train size: {ts}, mean test accuracy: {np.mean(accuracies)}, std test accuracy: {np.std(accuracies)}")

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    df.to_csv(RESULTS_DIR / f'{python_file_name_no_ext}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
