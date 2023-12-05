import pennylane as qml


def layer(x, params, wires, i0=0, inc=1):
    """
    Implements a layer the Hardware Efficient Ansatz (HEA) Embedding. Adapted from https://github.com/thubregtsen/qhack.

    Parameters
    ----------
    x : ndarray
        Input data for the layer.
    params : ndarray
        Trainable parameters for the layer.
        params[0, j] corresponds to the RY rotation angle for the j-th wire.
        params[1, j] corresponds to the CRZ entangling angle for the j-th wire pair.
    wires : list
        List of wire indices on which the gates are applied.
    i0 : int, optional
        Initial index for accessing the input data. Default is 0.
    inc : int, optional
        Increment for accessing the input data indices. Default is 1.

    Returns
    -------
    None
        The function modifies the quantum circuit in-place.

    Examples
    --------
    >>> x = np.array([0.1, 0.2, 0.3])
    >>> params = np.array([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    >>> wires = [0, 1, 2]
    >>> layer(x, params, wires)

    Notes
    -----
    This function defines a quantum circuit layer composed of Hadamard, RZ, RY,
    and CRZ gates. The layer acts on specified wires using input data `x` and trainable
    parameters `params`. The indices for accessing the input data are controlled by
    the parameters `i0` (initial index) and `inc` (increment).

    """
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring",
                  wires=wires, parameters=params[1])


def embedding(x, params, wires):
    """
    Quantum circuit embedding composed of multiple layers.  Adapted from https://github.com/thubregtsen/qhack.

    Parameters
    ----------
    x : ndarray
        Input data for the embedding.
    params : list of ndarrays
        List of trainable parameters for each layer in the embedding.
    wires : list
        List of wire indices on which the gates are applied.

    Returns
    -------
    None
        The function modifies the quantum circuit in-place.

    Examples
    --------
    >>> x = np.array([0.1, 0.2, 0.3])
    >>> params = [np.array([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]), np.array([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])]
    >>> wires = [0, 1, 2]
    >>> embedding(x, params, wires)

    Notes
    -----
    This function defines a quantum circuit embedding composed of multiple layers.
    Each layer a Hardware Efficient Ansatz (HEA) layer. The embedding acts on specified wires using input data `x` and a list of trainable parameters `params` for each layer.
    """
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))
