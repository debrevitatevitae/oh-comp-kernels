import numpy as np


def train(
    initial_params,
    loss_function,
    optimizer,
    num_epochs,
    dataloader,
    num_checkpoints=None,
):
    losses = []
    params = initial_params

    num_checkpoints = num_checkpoints or num_epochs

    epochs_to_checkpoint = num_epochs // num_checkpoints
    for ep in range(num_epochs):

        for X, y in dataloader:
            loss = loss_function(X, y, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        if (ep + 1) % epochs_to_checkpoint == 0:
            loss_at_checkpoint = np.mean(losses[-epochs_to_checkpoint:])
            print(f"Epoch {ep+1}, loss = {loss_at_checkpoint:.5f}")

    return params, losses
