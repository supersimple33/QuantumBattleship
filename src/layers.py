import pennylane as qml

from pennylane import numpy as np

import numpy as nnp
from pennylane.wires import WiresLike
from typing import Union

# import torch
import tensorflow as tf

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from functools import reduce

np_floats = Union[np.float16, np.float32, np.float64, np.longdouble]


def hots_to_sv(x: nnp.ndarray[np_floats]) -> nnp.ndarray[np_floats]:
    """Converts list of qubit rotations to a statevector"""
    assert len(x.shape) == 1, "x must be a 1D array"

    qc = QuantumCircuit(len(x))
    [qc.rx(nnp.pi * e, i) for i, e in enumerate(x)]
    sv = Statevector.from_instruction(qc).reverse_qargs()
    return sv.data.astype(nnp.complex64)


def convolution_op(params: np.ndarray[np_floats], wires: WiresLike) -> None:
    """Creates a convolution layer. Which consists of ry gates and entangling gates repeated according to the shape of the params"""
    assert len(wires) == params.shape[1], "params and wires must have the same length"
    assert type(wires) is list, "wires must be a list of wires"
    qml.BasicEntanglerLayers(
        params,
        wires,
        qml.RY,
    )


def pooling_op(params: np.ndarray[np_floats] | list[float], wires: WiresLike) -> None:
    """Creates a pooling layer. Which consists of 4 rotations a CNOT and a unrotation of the target qubit"""
    assert type(wires) is list, "wires must be a list of wires"

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
) -> None:
    """Creates a fully connected layer. Which consists of ry gates a cnot chain and then cnot gates connecting the ryed b and x wires"""
    assert (
        b_params.shape[0] == len(x_wires) == len(b_wires) == weight_params.shape[1]
    ), "params and wires must have the same length"
    assert (
        weight_params.shape[1] - 1 == weight_params.shape[0]
    ), "params must be a square matrix"
    assert type(x_wires) is list, "wires must be a list of wires"
    assert type(b_wires) is list, "wires must be a list of wires"

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


# MARK: - TensorFlow Stuff


def prob_extraction(x):
    return x[..., 1]


@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(tf.where(y_pred >= 0, 1.0, -1.0))
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))
