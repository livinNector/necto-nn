import numba as nb
from numba import njit, typed
from numba.experimental import jitclass
import numpy as np
from typing import List
from .activations import Activation, JitActivation
from .intializers import Initializer, XavierInit
# X - (batch_size, n_dims)
# W - (in_dims, out_dims)
# b - out_dims
# X @ W = (batch_size, out_dims)


@njit
def last_outer(X: np.ndarray, Y: np.ndarray):
    """Outer product along last axis"""
    m, n = X.shape[-1], Y.shape[-1]
    out = np.zeros(X.shape[:-1] + (m, n))
    for i in range(m):
        for j in range(n):
            out[..., i, j] = X[..., i] * Y[..., j]
    return out


class FeedForwardNetwork:
    def __init__(
        self,
        input_size: int,
        layers: list[list[int, Activation]],
        initializer: Initializer = None,
        weight_decay: float = 0,
    ):
        """Initializes a Feed forward neural network.

        Args:
            input_size (int): Number of inputs
            layers (list[tuple[int, Activation]]):
                List of tuples of layer sizes and activation functions.
                The last layer is the output layer.
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
        self.W, self.b = map(typed.List, self.initializer.init(self.layer_sizes))

        # outputs (activations) in each layer, O[0] is X
        self.O = typed.List([np.zeros((1, 1))] * (self.n_layers + 1))

        # gradients dW[i] = dO[i+1]/dW[i] and dW[L] is the gradient from loss
        self.dW = typed.List([np.zeros_like(W) for W in self.W])
        self.db = typed.List([np.zeros_like(b) for b in self.b])

    def forward(self, X):
        """Forward pass returning the output."""
        self.O[0] = X

        for i, activation in enumerate(self.activations, 0):
            self.O[i + 1] = activation.forward(np.dot(self.O[i], self.W[i]) + self.b[i])
        return self.O[-1]

    def backward(self, gradient):
        """Backward pass computing the required gradients."""

        for i in range(self.n_layers - 1, -1, -1):
            activation = self.activations[i]
            act_gradient = activation.backward(gradient, self.O[i + 1])
            # (batch, out) , (batch, in) -> (batch, in, out) and mean
            #
            self.dW[i][...] = (
                last_outer(self.O[i], act_gradient).sum(axis=0) / act_gradient.shape[0]
            )
            self.db[i][...] = act_gradient.sum(axis=0) / act_gradient.shape[0]
            if i != 0:
                # (batch, out) , (in, out) -> (batch, in)
                gradient = np.dot(act_gradient, self.W[i].T)
                # gradient = np.einsum("bo,io->bi", act_gradient, self.W[i])

        if self.weight_decay > 0:
            for i in range(len(self.W)):
                self.dW[i] += 2 * self.weight_decay * self.W[i]

    def get_compiled(self):
        # initialize the model
        self.init()

        activations_list = typed.List([act.compiled() for act in self.activations])
        # array2d_type = types.Array(types.float64, 2, "C")
        # array1d_type = types.Array(types.float64, 1, "C")
        return CompiledFeedForwardNetwork(
            self.input_size,
            activations_list,
            self.weight_decay,
            self.n_layers,
            self.W,
            self.b,
            self.O,
            self.dW,
            self.db,
        )


@jitclass
class CompiledFeedForwardNetwork(FeedForwardNetwork):
    input_size: int
    activations: List[JitActivation.class_type.instance_type]
    weight_decay: float
    n_layers: int
    W: List[nb.float64[:, ::1]]
    b: List[nb.float64[::1]]
    O: List[nb.float64[:, ::1]]
    dW: List[nb.float64[:, ::1]]
    db: List[nb.float64[::1]]

    def __init__(
        self,
        input_size,
        activations,
        weight_decay,
        n_layers,
        W,
        b,
        O,
        dW,
        db,
    ):
        self.input_size = input_size
        self.activations = activations
        self.weight_decay = weight_decay
        self.n_layers = n_layers
        self.W = W
        self.b = b
        self.O = O
        self.dW = dW
        self.db = db

    def init(self):
        pass

    def get_compiled(self):
        pass
