import os
import argparse
from typing import List
from utils.counterfactual_data_utils import create_single_source_counterfactual_dataset
from utils.das_utils import CNNConfig, CustomLowRankRotatedSpaceIntervention, das_evaluate, das_train
from utils.model_utils import PyTorchCNN
import pyvene as pv
import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import itertools
from tqdm import tqdm
from pyvene import type_to_dimension_mapping

random.seed(42)
np.random.seed(42)
_ = torch.manual_seed(42)
_ = torch.cuda.manual_seed(42)

# hack to add custom model to pyvene config
type_to_dimension_mapping[PyTorchCNN] = {
    'conv1': str(28 * 28 * 16),
    'conv2': str(14 * 14 * 32),
    'conv3': str(7 * 7 * 64),
    'fc1': '3'
}

LAYERS = {
    'conv1': 28 * 28 * 16,
    'conv2': 14 * 14 * 32,
    'conv3': 7 * 7 * 64,
    'fc1': 3
}

COEFFICIENTS = [
    np.array([0.4, 0.4, 0.4]),
    np.array([0.3, 0.4, 0.5]),
    np.array([0.5, 0.4, 0.3]),
]

# 7 * 4 * 300 = 8400 experiments * 60 seconds = 504000 seconds / 3600 = 140 hours / 5 splits = 28 hours

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

def das_experiment(
    # experiment params
    model : PyTorchCNN,
    variables : List[int],
    layer : str,
    # data params
    images : np.ndarray,
    labels : np.ndarray,
    coefficients : np.ndarray,
    n_train : int = 10000,
    n_eval : int = 1000,
    # das params
    intervention_size : int = 4,
    lr : float = 0.0005,
    num_epochs : int = 3,
    batch_size : int = 512,
    device : str = 'cuda'
):
    """
    Run a DAS experiment to localize the variables in the given layer of the model.
    """
    assert layer in LAYERS, f"Invalid layer: {layer}"
    assert variables in VARIABLE_PARTITIONS, f"Invalid variables: {variables}"

    model.config = CNNConfig(
        hidden_size=LAYERS[layer],
    )

    # set up intervention config (1 dim for last layer)
    if layer == 'fc1':
        intervention_size = 1
    representations = [{
        "component": f"{layer}.output",
        "low_rank_dimension": intervention_size,
    }]

    pv_config = pv.IntervenableConfig(
        representations=representations,
        intervention_types=CustomLowRankRotatedSpaceIntervention
    )
    pv_model = pv.IntervenableModel(pv_config, model)
    pv_model.set_device(device)

    X_base_train, X_sources_train, _, _, y_counterfactual_train = create_single_source_counterfactual_dataset(
        variables, images, labels, coefficients, size=n_train
    )

    das_train(
        pv_model, 
        X_base_train, X_sources_train, y_counterfactual_train, 
        lr=lr, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        device=device
    )

    X_base_eval, X_sources_eval, _, _, y_counterfactual_eval = create_single_source_counterfactual_dataset(
        variables, images, labels, coefficients, size=n_eval
    )

    accuracy = das_evaluate(
        pv_model, 
        X_base_eval, 
        X_sources_eval, 
        y_counterfactual_eval,
        batch_size=batch_size,
        device=device
    )
    return accuracy

def main(
    model_indices: List[int],
    n_train: int = 10000,
    n_eval: int = 1000,
    intervention_size: int = 4,
    lr: float = 0.0005,
    num_epochs: int = 3,
    batch_size: int = 512,
    device: str = 'cuda'
):
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')

    results = []
    length = len(model_indices) * len(VARIABLE_PARTITIONS) * len(LAYERS)
    for model_index, variables, layer in tqdm(itertools.product(model_indices, VARIABLE_PARTITIONS, LAYERS), total=length, desc='Experiments'):
        # load model
        model = PyTorchCNN()
        model.load_state_dict(torch.load(f'pytorch_models/cnn_model_{model_index + 1}.pth'))

        coefficients = COEFFICIENTS[model_index % len(COEFFICIENTS)]

        print('RUNNING EXPERIMENT:', model_index + 1, variables, layer)

        # run experiment
        accuracy = das_experiment(
            model=model,
            variables=variables,
            layer=layer,
            images=images,
            labels=labels,
            coefficients=coefficients,
            n_train=n_train,
            n_eval=n_eval,
            intervention_size=intervention_size,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        results.append({
            'model': model,
            'variables': variables,
            'layer': layer,
            'accuracy': accuracy
        })
    results_df = pd.DataFrame(results)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(f'results/results_{timestamp}.csv')

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=1)
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_eval', type=int, default=1000)
    parser.add_argument('--intervention_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"Running DAS experiments for models {args.start_index} to {args.end_index}")
    # run main
    main(
        model_indices=list(range(args.start_index, args.end_index)),
        n_train=args.n_train,
        n_eval=args.n_eval,
        intervention_size=args.intervention_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device
    )