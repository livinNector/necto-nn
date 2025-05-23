import numpy as np


class Initializer:
    def init(self, layer_sizes: list[int]):
        """Initialize a list of weights and biases.

        Args:
            layer_sizes (list[int]) :
                List of layer sizes including input and output layers.

        """
        raise NotImplementedError


class RandomInit(Initializer):
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def init(self, layer_sizes: list[int]):
        weights = [
            np.random.uniform(self.low, self.high, size=(n_i, n_o))
            for n_i, n_o in zip(layer_sizes, layer_sizes[1:])
        ]
        biases = [np.zeros(n_o) for n_o in layer_sizes[1:]]
        return weights, biases


class XavierInit(Initializer):
    def init(self, layer_sizes: list[int]):
        weights = [
            np.random.uniform(x := -np.sqrt(6 / (n_i + n_o)), -x, size=(n_i, n_o))
            for n_i, n_o in zip(layer_sizes, layer_sizes[1:])
        ]
        biases = [np.zeros(n_o) for n_o in layer_sizes[1:]]
        return weights, biases


initalizers = {"random": RandomInit, "xavier": XavierInit}


def get_initializer(name):
    return initalizers[name]
