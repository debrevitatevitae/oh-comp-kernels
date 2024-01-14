import pennylane as qml


def layer(x, params, wires, i0=0, inc=1):
    """
    Apply a layer of quantum operations to a set of qubits.

    Args:
        x (array-like): Input vector.
        params (array-like): Array-like object containing the parameters for the quantum operations.
        wires (list): List of qubits to apply the operations on.
        i0 (int): Starting index for the input values. Default is 0.
        inc (int): Increment value for the input index. Default is 1.

    Returns:
        None
    """
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])  # modulo to reuse the same x
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring",
                  wires=wires, parameters=params[1])


def trainable_embedding(x, params, wires):
    """Adapted from https://github.com/thubregtsen/qhack"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))
