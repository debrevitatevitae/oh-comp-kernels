import os
import pickle
import time

import jax
import jaxopt
from jax import numpy as jnp
import numpy as np
import optax
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler

from project_directories import PICKLE_DIR, RESULTS_DIR
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

    # create pandas DataFrame to store results
    columns = ["run_id", "epoch", "kta"]
    df = pd.DataFrame(columns=columns)

    # define the embedding kernel (with JAX)
    num_qubits = 6
    num_layers = 4
    wires = list(range(num_qubits))
    dev = qml.device('default.qubit.jax', wires=num_qubits)

    @jax.jit
    @qml.qnode(device=dev, interface="jax")
    def kernel(x1, x2, params):
        """x1, x2 and params must be JAX arrays"""
        embedding(x1, params, wires)
        qml.adjoint(embedding)(x2, params, wires)
        return qml.expval(qml.Projector([0]*num_qubits, wires=wires))

    def kta_loss(params, X, y):
        # compute the kernel matrix
        N = len(y)
        K = jnp.eye(N)
        for i in range(N-1):
            for j in range(i+1, N):
                # Compute kernel value and fill in both (i, j) and (j, i) entries
                K_i_j = kernel(X[i], X[j], params)
                K = K.at[i, j].set(K_i_j)
                K = K.at[j, i].set(K_i_j)

        # compute the target kernel
        T = jnp.outer(y, y)

        # compute polarity
        polarity = jnp.sum(K * T)

        # normalise
        kta = polarity / (N * jnp.sqrt(jnp.sum(K * K)))

        return -kta

    # optimizer
    eta = 0.01
    adam = optax.adam(learning_rate=eta)
    opt = jaxopt.OptaxSolver(opt=adam, fun=kta_loss)

    # batching function. Keep the batch size very low to avoid memory issues of the jaxopt optimizer
    def batch_data(key, batch_size=5):
        _, key = jax.random.split(key)
        idxs_batch = jax.random.choice(key, num_samples, shape=(batch_size,))
        return key, X_train_scaled[idxs_batch], y_train[idxs_batch]

    # function to compute the average KTA loss every N epochs
    num_batches_loss_eval = 500
    batch_size_loss_eval = 10

    def compute_ave_kta_loss(key, params):
        ave_loss = 0.
        for _ in range(num_batches_loss_eval):
            key, X_batch, y_batch = batch_data(
                key, batch_size=batch_size_loss_eval)
            ave_loss += 1/num_batches_loss_eval * \
                kta_loss(params, X_batch, y_batch)
        return key, ave_loss

    # initial parameters and optimizer initialization
    num_runs = 5
    num_epochs = 500
    epochs_per_checkpoint = 50

    best_kta = 0.
    best_run = None
    best_params = None

    for run in range(num_runs):
        print("New optimization run")

        _, key = jax.random.split(key)
        init_params = jax.random.uniform(
            key, minval=0., maxval=2*jnp.pi, shape=(num_layers, 2, num_qubits))

        key, X_batch, y_batch = batch_data(key)
        opt_state = opt.init_state(init_params, X_batch, y_batch)
        params = init_params

        # optimization loop with early stopping

        for ep in range(num_epochs):
            # Every some epochs report loss value averaged over some batches
            if ep % epochs_per_checkpoint == 0 or ep+1 == num_epochs:
                key, loss = compute_ave_kta_loss(key, params)
                kta = -loss
                print(
                    f"Epoch {ep}: kta averaged over {num_batches_loss_eval} batches = {kta:.4f}", flush=True)

                if kta > best_kta:
                    best_kta = kta
                    best_params = params
                    best_run = run

                df.loc[len(df)] = {
                    "run_id": run,
                    "epoch": ep,
                    "kta": kta
                }

            # select a batch
            key, X_batch, y_batch = batch_data(key)

            # optimization step
            params, opt_state = opt.update(
                params, opt_state, X_batch, y_batch)

    # at the end store the best run and best kta with a 'trick'
    df.loc[len(df)] = {
        "run_id": best_run,
        "epoch": -1,  # convention value: indicates best kta
        "kta": best_kta
    }

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    df.to_csv(RESULTS_DIR /
              f'{python_file_name_no_ext}_0{num_qubits}0{num_layers}.csv', index=False)

    # pickle the best parameters
    with open(PICKLE_DIR / f'{python_file_name_no_ext}_0{num_qubits}0{num_layers}.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
