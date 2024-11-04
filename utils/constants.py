import numpy as np

LAYERS = {
    # 'conv1': 28 * 28 * 16,
    'conv2': 14 * 14 * 32,
    'conv3': 7 * 7 * 64,
    'fc1': 3
}

COEFFICIENTS = [
    np.array([0.4, 0.4, 0.4]),
    np.array([0.3, 0.4, 0.5]),
    np.array([0.5, 0.4, 0.3]),
]

# ALL SINGLE-SOURCE ??
VARIABLE_PARTITIONS = [
    [0, 1, 2],
    [0, 1],
    [0, 2],
    [1, 2],
    [0],
    [1],
    [2]
]