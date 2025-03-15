import pickle
from typing import List

import numba as nb
import numpy as np
from numba import njit, typed
from numba.experimental import jitclass

from .activations import Activation, JitActivation
from .intializers import Initializer, XavierInit

# X - (batch_size, n_dims)
# W - (in_dims, out_dims)
# b - out_dims
# X @ W = (batch_size, out_dims)


@njit
def last_outer(a: np.ndarray, b: np.ndarray):
    """Outer product along last axis."""
    m, n = a.shape[-1], b.shape[-1]
    out = np.zeros(a.shape[:-1] + (m, n))
    for i in range(m):
        for j in range(n):
            out[..., i, j] = a[..., i] * b[..., j]
    return out


class FeedForwardNetwork:
    def __init__(
        self,
        input_size: int,
        layers: list[list[int, Activation]],
        initializer: Initializer = None,
        weight_decay: float = 0,
    ):
        """Feed forward neural network.

        Args:
            input_size (int): Number of inputs
            layers (list[tuple[int, Activation]]):
                List of tuples of layer sizes and activation functions.
                The last layer is the output layer.
            initializer (Initializer): weight initializer (random or xavier)
            weight_decay (float): weight decay parameter.

        """
        self.input_size = input_size
        self.layers = layers
        self.weight_decay = weight_decay

        if initializer is None:
            initializer = XavierInit()

        self.initializer = initializer

    def init(self):
        self.n_layers = len(self.layers)
        self.layer_sizes = [self.input_size, *(x for x, _ in self.layers)]
        self.activations = [x for _, x in self.layers]

        # weights and biases W[0] to W[L-1]
        self.weights, self.biases = map(
            typed.List, self.initializer.init(self.layer_sizes)
        )

        # outputs after activation function in each layer, O[0] is X
        self.outputs = typed.List([np.zeros((1, 1))] * (self.n_layers + 1))

        # gradients dW[i] = dO[i+1]/dW[i] and dW[L] is the gradient from loss
        self.d_weights = typed.List([np.zeros_like(W) for W in self.weights])
        self.d_biases = typed.List([np.zeros_like(b) for b in self.biases])

    def forward(self, X):
        """Forward pass returning the output."""
        self.outputs[0] = X

        for i, activation in enumerate(self.activations, 0):
            self.outputs[i + 1] = activation.forward(
                np.dot(self.outputs[i], self.weights[i]) + self.biases[i]
            )
        return self.outputs[-1]

    def backward(self, gradient):
        """Backward pass computing the required gradients."""
        for i in range(self.n_layers - 1, -1, -1):
            activation = self.activations[i]
            act_gradient = activation.backward(gradient, self.outputs[i + 1])
            # (batch, out) , (batch, in) -> (batch, in, out) and mean
            #
            self.d_weights[i][...] = (
                last_outer(self.outputs[i], act_gradient).sum(axis=0)
                / act_gradient.shape[0]
            )
            self.d_biases[i][...] = act_gradient.sum(axis=0) / act_gradient.shape[0]
            if i != 0:
                # (batch, out) , (in, out) -> (batch, in)
                gradient = np.dot(act_gradient, self.weights[i].T)
                # gradient = np.einsum("bo,io->bi", act_gradient, self.W[i])

        if self.weight_decay > 0:
            for i in range(len(self.weights)):
                self.d_weights[i] += 2 * self.weight_decay * self.weights[i]

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def get_compiled(self):
        # initialize the model
        self.init()

        activations_list = typed.List([act.compiled() for act in self.activations])

        return CompiledFeedForwardNetwork(
            self.input_size,
            activations_list,
            self.weight_decay,
            self.n_layers,
            self.weights,
            self.biases,
            self.outputs,
            self.d_weights,
            self.d_biases,
        )


@jitclass
class CompiledFeedForwardNetwork(FeedForwardNetwork):
    input_size: int
    activations: List[JitActivation.class_type.instance_type]
    weight_decay: float
    n_layers: int
    weights: List[nb.float64[:, ::1]]
    biases: List[nb.float64[::1]]
    outputs: List[nb.float64[:, ::1]]
    d_weights: List[nb.float64[:, ::1]]
    d_biases: List[nb.float64[::1]]

    def __init__(
        self,
        input_size,
        activations,
        weight_decay,
        n_layers,
        weights,
        biases,
        outputs,
        d_weights,
        d_biases,
    ):
        self.input_size = input_size
        self.activations = activations
        self.weight_decay = weight_decay
        self.n_layers = n_layers
        self.weights = weights
        self.biases = biases
        self.outputs = outputs
        self.d_weights = d_weights
        self.d_biases = d_biases

    def init(self):
        pass

    def get_compiled(self):
        pass
