import numpy as np
from .activations import Activation
from .intializers import Initializer, XavierInit

# X - (batch_size, n_dims)
# W - (in_dims, out_dims)
# b - out_dims
# X @ W = (batch_size, out_dims)


class FeedForwardNetwork:
    def __init__(
        self,
        input_size: int,
        layers: list[tuple[int, Activation]],
        initializer: Initializer = None,
        weight_decay: float = None,
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

        self.layer_sizes = [input_size] + [x for x, _ in layers]
        self.n_hidden_layers = len(layers)

        if initializer is None:
            initializer = XavierInit()

        self.initializer = initializer

    def init(self):
        # weights and biases W[0] to W[L-1]
        self.W, self.b = self.initializer.init(layer_sizes=self.layer_sizes)

        # outputs (activations) in each layer, O[0] is X
        self.O = [None] * (self.n_hidden_layers + 1)

        # gradients dW[i] = dO[i+1]/dW[i] and dW[L] is the gradient from loss
        self.dW = [None] * (self.n_hidden_layers)
        self.db = [None] * (self.n_hidden_layers)

    def forward(self, X):
        """Forward pass returning the output."""
        self.O[0] = X
        for i, (_, activation) in enumerate(self.layers):
            self.O[i + 1] = activation(self.O[i] @ self.W[i] + self.b[i])

        return self.O[-1]

    def backward(self, gradient):
        """Backward pass computing the required gradients."""
        for i, (_, activation) in reversed(list(enumerate(self.layers))):
            act_gradient = activation.backward(gradient, self.O[i + 1])
            # (batch, out) , (batch, in) -> (batch, in, out)
            self.dW[i] = np.einsum("bo,bi->bio", act_gradient, self.O[i]).mean(axis=0)
            self.db[i] = act_gradient.mean(axis=0)
            if i != 0:
                # (batch, out) , (in, out) -> (batch, in)
                gradient = np.einsum("bo,io->bi", act_gradient, self.W[i])

        if self.weight_decay:
            for i in range(len(self.W)):
                self.dW[i] += 2 * self.weight_decay * self.W[i]
