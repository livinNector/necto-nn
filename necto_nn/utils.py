from functools import wraps

import numpy as np
from numpy.core.einsumfunc import _parse_einsum_input


def array_args(func):
    """Wrapper to ensure the arguments to the function are np arrays."""

    @wraps(func)
    def inner(*args):
        return func(
            *map(lambda x: x if isinstance(x, np.ndarray) else np.array(x), args)
        )

    return inner


def one_hot(y, n=None):
    if not n:
        n = np.unique(y).shape[0]
    return y[..., np.newaxis] == np.arange(n).reshape(1, -1)


def flatten(x):
    return x.reshape(x.shape[0], -1)


def einsum(*operands, **kwargs):
    """Einsum with sum over ellipses."""
    # https://github.com/numpy/numpy/issues/9984
    # einsum to support summing over ellipses
    operands = _parse_einsum_input(operands)
    return np.einsum("->".join(operands[:-1]), *operands[-1], **kwargs)


def train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=True):
    """Split dataset into training and testing sets.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)

    """
    n = x.shape[0]
    np.random.seed(random_state)
    if isinstance(test_size, float):
        test_size = int(n * test_size)
    train_size = n - test_size

    if stratify:
        unique_classes = np.unique(y)
        test_indices = []
        train_indices = []

        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]

            if shuffle:
                np.random.shuffle(class_indices)

            split_idx = int(len(class_indices) * (train_size / n))
            train_indices.extend(class_indices[:split_idx])
            test_indices.extend(class_indices[split_idx:])
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if shuffle:  # to shuffle different classes
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
    else:
        indices = np.arange(len(x))
        if shuffle:
            np.random.shuffle(indices)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test
