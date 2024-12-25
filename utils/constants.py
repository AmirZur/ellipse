NUM_MODELS = 20

LAYERS = {
    'conv1': 28 * 28 * 16,
    'conv2': 14 * 14 * 32,
    'conv3': 7 * 7 * 64,
    'fc1': 3
}

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

DATA_DIR = 'results_transfer_ellipse/results_12_6'
MODEL_DIR = 'pytorch_models'
DAS_RESULTS_DIR = 'results_das_ellipse'
PROBE_RESULTS_DIR = 'results_probe_ellipse'