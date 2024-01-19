import os
import time

import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler

from project_directories import RESULTS_DIR
from ohqk.utils import load_split_data


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


def frobenius_ip(A, B):
    return np.sum(A * B)


if __name__ == '__main__':
    start = time.time()

    np.random.seed(42)
    key = jax.random.PRNGKey(42)

    # data loading, splitting, and scaling
    X_train, X_test, y_train, y_test = load_split_data(test_size=0.2)
    N = len(y_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)

    # create pandas DataFrame to store results
    columns = ["width", "depth", "mean_kta", "var_kta", "min_kta", "max_kta"]
    df = pd.DataFrame(columns=columns)

    # define the search parameters for kernel selection
    width_grid = [3, 6, 9, 12]
    depth_grid = [1, 2, 3, 4]
    num_batches = 500
    batch_size = 10
    num_param_samples = 10

    # batching function
    def batch_data_and_labels():
        idxs_batch = np.random.choice(N, size=batch_size)
        return X_train_scaled[idxs_batch], y_train[idxs_batch]

    # kernel selection loop
    for width in width_grid:

        # device
        dev = qml.device("default.qubit.jax", wires=width)
        wires = list(range(width))

        # define the embedding kernel (with JAX)
        @jax.jit
        @qml.qnode(device=dev, interface="jax")
        def kernel(x1, x2, params):
            """x1, x2 and params must be JAX arrays"""
            embedding(x1, params, wires)
            qml.adjoint(embedding)(x2, params, wires)
            return qml.expval(qml.Projector([0]*width, wires=wires))

        # square kernel matrix function
        def square_kernel_matrix(X, params):
            N = X.shape[0]
            K = np.eye(N)

            # convert X to JAX array
            X = jnp.array(X)

            for i in range(N-1):
                for j in range(i+1, N):
                    # Compute kernel value and fill in both (i, j) and (j, i) entries
                    K_i_j = kernel(X[i], X[j], params)
                    K[i, j] = K_i_j
                    K[j, i] = K_i_j

            return K

        # KTA function
        def target_alignment(X, Y, params, rescale_class_labels=False):

            N = X.shape[0]

            if rescale_class_labels:
                nplus = np.count_nonzero(Y == 1)
                nminus = len(Y) - nplus
                _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
            else:
                _Y = np.array(Y)

            K = square_kernel_matrix(X, params)

            K_target = np.outer(_Y, _Y)
            return frobenius_ip(K, K_target) / (N * np.sqrt(frobenius_ip(K, K)))

        for depth in depth_grid:

            ktas_current_model = []

            for _ in range(num_param_samples):
                # initialize variational parameters at random
                params = jax.random.uniform(key, shape=(
                    depth, 2, width), minval=0., maxval=2*np.pi)
                _, key = jax.random.split(key)

                kta = 0.
                for _ in range(num_batches):
                    X_train_scaled_batch, y_train_batch = batch_data_and_labels()

                    kta += 1/num_batches * \
                        target_alignment(X_train_scaled_batch,
                                         y_train_batch, params)

                ktas_current_model.append(kta)

            max_kta = np.max(ktas_current_model)
            var_kta = np.var(ktas_current_model)

            df.loc[len(df)] = {
                "width": width,
                "depth": depth,
                "mean_kta": np.mean(ktas_current_model),
                "var_kta": var_kta,
                "min_kta": np.min(ktas_current_model),
                "max_kta": max_kta,
            }

            print(
                f"Width = {width:d}, depth = {depth:d}: max KTA = {max_kta:.4f}, var KTA = {var_kta:.4f}")

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    df.to_csv(RESULTS_DIR / f'{python_file_name_no_ext}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
