import pennylane as qml

from pennylane import numpy as np

# import numpy as np
from pennylane.wires import WiresLike
from typing import Union
import torch
import tensorflow as tf

np_floats = Union[np.float16, np.float32, np.float64, np.longdouble]

# def convolution_layer(params: np.ndarray[np_floats], wires: WiresLike) -> None:
#     """Creates a convolution layer. Which consists of ry gates and entangling gates repeated according to the shape of the params"""
#     assert len(params.shape) == 2, "params must be a 2D array"
#     assert len(wires) == params.shape[1], "params and wires must have the same length"

#     def _layer(params: np.ndarray) -> None:
#         for wire in wires:
#             qml.RY(params[wire], wires=wire)

#         for i in range(len(wires) - 1):
#             qml.CNOT(wires=[wires[i], wires[-1]])

#     qml.layer(
#         _layer,
#         params.shape[0],
#         params,
#     )


def convolution_op(params: np.ndarray[np_floats], wires: WiresLike) -> None:
    """Creates a convolution layer. Which consists of ry gates and entangling gates repeated according to the shape of the params"""
    assert len(wires) == params.shape[1], "params and wires must have the same length"
    assert wires is not torch.Tensor, "wires must be a list of wires"
    qml.BasicEntanglerLayers(
        params,
        wires,
        qml.RY,
    )


def pooling_op(params: np.ndarray[np_floats] | list[float], wires: WiresLike) -> None:
    """Creates a pooling layer. Which consists of 4 rotations a CNOT and a unrotation of the target qubit"""
    assert wires is not torch.Tensor, "wires must be a list of wires"

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
    conv_params = np.reshape(
        conv_params, (conv_params.shape[0], conv_params.shape[1] * conv_params.shape[2])
    )
    pool_params = np.ravel(pool_params)

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
                        .take(range(j + l, j + KERNEL_SIZE + k), mode="wrap", axis=1)
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
        len(b_params) == len(x_wires) == len(b_wires) == weight_params.shape[1]
    ), "params and wires must have the same length"
    assert (
        weight_params.shape[1] - 1 == weight_params.shape[0]
    ), "params must be a square matrix"
    assert x_wires is not torch.Tensor, "wires must be a list of wires"
    assert b_wires is not torch.Tensor, "wires must be a list of wires"

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


class PatchedKerasLayer(qml.qnn.KerasLayer):
    def call(self, inputs):
        """Evaluates the QNode on input data using the initialized weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        inputs = tf.transpose(inputs, perm=(1, 2, 0))

        # calculate the forward pass as usual
        results = self._evaluate_qnode(inputs)

        # reshape to the correct number of batch dims
        # if has_batch_dim:
        #     # pylint:disable=unexpected-keyword-arg,no-value-for-parameter
        #     new_shape = tf.concat([batch_dims, tf.shape(results)[1:]], axis=0)
        #     results = tf.reshape(results, new_shape)

        return results

    def _evaluate_qnode(self, x):
        """Evaluates a QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {
            **{self.input_arg: x},
            **{k: 1.0 * w for k, w in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, (list, tuple)):
            # multi-return and no batch dim
            return tf.transpose(tf.convert_to_tensor(res), perm=(1, 0, 2))

        return res


# MARK: - PyTorch Stuff


class ProbExtractionLayer(torch.nn.Module):
    def __init__(self):
        super(ProbExtractionLayer, self).__init__()

    # def forward(self, x):
    #     # assert x.shape[-1] == 2, "The last dimension must have size 2"
    #     return x[..., 1]

    def forward(self, x):
        # Convert expectation values to probabilities of measuring 1
        return (x + 1) / 2


class PatchedTorchLayer(qml.qnn.TorchLayer):
    """this patch allows us to use 2d inputs and weights"""

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """

        # in case the input has more than one batch dimension

        # calculate the forward pass as usual
        results = self._evaluate_qnode(inputs)

        if isinstance(results, tuple):
            return torch.stack(results, dim=0)

        # reshape to the correct number of batch dims

        return results

    def _evaluate_qnode(self, x):
        """Evaluates the QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)

        return torch.hstack(res).type(x.dtype)
