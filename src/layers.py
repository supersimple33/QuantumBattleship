import pennylane as qml

from pennylane import numpy as np

import numpy as nnp
from pennylane.wires import WiresLike
from typing import Union

# import torch
import tensorflow as tf

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from functools import reduce
from itertools import chain

np_floats = Union[np.float16, np.float32, np.float64, np.longdouble]


def hots_to_sv(x: nnp.ndarray[np_floats]) -> nnp.ndarray[np_floats]:
    """Converts list of qubit rotations to a statevector"""
    assert len(x.shape) == 1, "x must be a 1D array"

    qc = QuantumCircuit(len(x))
    [qc.rx(nnp.pi * e, i) for i, e in enumerate(x)]
    sv = Statevector.from_instruction(qc).reverse_qargs()
    return sv.data.astype(nnp.complex64)


def generate_esv(*x: int, horiz: bool, noise=0.05):
    """Color the x1 and x2 rows or columns of a 4x4 grid and then apply noise"""
    assert 1 <= len(x) <= 3, "x must be a list of 1 to 4 integers"
    assert all(0 <= i < 4 for i in x), "x must be a list of integers between 0 and 3"
    assert len(set(x)) == len(x), "x must be a list of unique integers"

    qc = QuantumCircuit(16)
    # apply hadamard to a qubit which isn't active
    xs = [
        [i + xi * 4 for i in range(4)] if horiz else [xi + i * 4 for i in range(4)]
        for xi in x
    ]
    unused = list(set(range(16)) - set(chain(*xs)))[:4]

    # entangle the two possible rows and or columns
    # uniform = sum(Statevector.from_int(i, dims=2**4) for i in range(len(x)))
    uniform = reduce(
        lambda a, b: a + b,
        (Statevector.from_int(i, dims=2**4) for i in range(len(x))),
    )
    qc.append(
        StatePreparation(uniform, normalize=True),
        unused,
    )
    [[qc.mcx(unused, j, ctrl_state=i) for j in xis] for i, xis in enumerate(xs)]
    qc.append(
        StatePreparation(uniform, normalize=True, inverse=True),
        unused,
    )

    # apply noise
    [qc.rx(nnp.random.normal(0, noise * nnp.pi), i) for i in range(16)]

    sv = Statevector.from_instruction(qc).reverse_qargs()
    return sv.data.astype(nnp.complex64)


def convolution_op(params: np.ndarray[np_floats], wires: WiresLike) -> None:
    """Creates a convolution layer. Which consists of ry gates and entangling gates repeated according to the shape of the params"""
    assert len(wires) == params.shape[1], "params and wires must have the same length"
    isinstance(wires, list | nnp.ndarray), "wires must be a list of wires"
    qml.BasicEntanglerLayers(
        params,
        wires,
        qml.RY,
    )


def pooling_op(params: np.ndarray[np_floats] | list[float], wires: WiresLike) -> None:
    """Creates a pooling layer. Which consists of 4 rotations a CNOT and a unrotation of the target qubit"""
    assert isinstance(wires, list | nnp.ndarray), "wires must be a list of wires"

    for i, wire in enumerate(wires):
        qml.RY(params[i], wires=wire)
    for i in range(len(wires) - 1):
        qml.CNOT(
            wires=[wires[i], wires[-1]]
        )  # could also do a chain ie basic entangler
    qml.RY(-params[-1], wires=wires[-1])


def convolution_pooling_op(
    conv_params: np.ndarray[np_floats],
    pool_params: np.ndarray[np_floats],
    wire_arr: np.ndarray,
    STRIDE: int,
) -> None:
    KERNEL_SIZE = conv_params.shape[1]
    N = wire_arr.shape[0]
    conv_params = tf.reshape(
        conv_params, (conv_params.shape[0], conv_params.shape[1] * conv_params.shape[2])
    )
    pool_params = tf.keras.ops.ravel(pool_params)

    # Convolution layer
    for k in range(0, KERNEL_SIZE, STRIDE):
        for l in range(0, KERNEL_SIZE, STRIDE):
            for i in range(0, N, KERNEL_SIZE):
                for j in range(0, N, KERNEL_SIZE):
                    convolution_op(
                        conv_params,
                        wire_arr.take(
                            range(i + k, i + KERNEL_SIZE + k), mode="wrap", axis=0
                        )
                        .take(range(j + l, j + KERNEL_SIZE + l), mode="wrap", axis=1)
                        .flatten()
                        .tolist(),
                    )

    # Pooling layer
    for i in range(0, N, KERNEL_SIZE):
        for j in range(0, N, KERNEL_SIZE):
            pooling_op(
                pool_params,
                wire_arr[i : i + KERNEL_SIZE, j : j + KERNEL_SIZE].flatten().tolist(),
            )


def fully_connected_op(
    weight_params: np.ndarray[np_floats],
    b_params: np.ndarray[np_floats],
    x_wires: WiresLike,
    b_wires: WiresLike,
    uncompute: bool = False,
) -> None:
    """Creates a fully connected layer. Which consists of ry gates a cnot chain and then cnot gates connecting the ryed b and x wires"""
    assert (
        b_params.shape[0] == len(x_wires) == len(b_wires) == weight_params.shape[1]
    ), "params and wires must have the same length"
    assert (
        weight_params.shape[1] - 1 == weight_params.shape[0]
    ), "params must be a square matrix"
    assert isinstance(x_wires, list | nnp.ndarray), "wires must be a list of wires"
    assert isinstance(b_wires, list | nnp.ndarray), "wires must be a list of wires"

    for i in range(weight_params.shape[0]):
        for j in range(len(x_wires)):
            qml.RY(weight_params[i, j], wires=x_wires[j])
        for j in range(len(x_wires)):
            qml.CNOT(
                wires=[x_wires[j % len(x_wires)], x_wires[(i + j + 1) % len(x_wires)]]
            )

    for i in range(len(b_wires)):
        qml.RY(b_params[i], wires=b_wires[i])
    for i in range(len(b_wires)):
        qml.CNOT(wires=[b_wires[i], x_wires[i]])
    if uncompute:
        for i in range(len(b_wires)):
            qml.RY(-b_params[i], wires=b_wires[i])


def compressed_fc_op(
    weight_params: np.ndarray[np_floats],
    b_params: np.ndarray[np_floats],
    x_wires: WiresLike,
    b_wires: WiresLike,
) -> None:
    """Creates a compressed fully connected layer. Just as the last layer except that it does not have full b wires"""
    assert weight_params.shape[1] == len(
        x_wires
    ), "params and wires must have the same length"
    # assert len(b_wire) == 1, "b_wire must be a single wire"
    assert len(b_params.shape) == 1, "b_params must be a 1D array"
    assert b_params.shape[0] >= len(
        b_wires
    ), "b_params must be at least as long as b_wires"
    assert (
        len(x_wires) >= b_params.shape[0]
    ), "x_wires must be at least as long as b_params"
    assert isinstance(x_wires, list | nnp.ndarray), "wires must be a list of wires"
    assert isinstance(b_wires, list | nnp.ndarray), "wires must be a list of wires"

    for i in range(weight_params.shape[0]):
        for j in range(len(x_wires)):
            qml.RY(weight_params[i, j], wires=x_wires[j])
        for j in range(len(x_wires)):
            qml.CNOT(
                wires=[x_wires[j % len(x_wires)], x_wires[(i + j + 1) % len(x_wires)]]
            )

    for i in range(0, b_params.shape[0], len(b_wires)):
        window = list(range(i, min(i + len(b_wires), b_params.shape[0])))
        for k_wire, j in enumerate(window):
            qml.RY(b_params[j], wires=b_wires[k_wire])
            qml.CNOT(wires=[b_wires[k_wire], x_wires[j]])
            qml.RY(-b_params[j], wires=b_wires[k_wire])


# MARK: - TensorFlow Stuff


def prob_extraction(x):
    return x[..., 1]


@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(tf.where(y_pred >= 0, 1.0, -1.0))
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))
