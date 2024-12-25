import os
import argparse
from typing import List
from utils.counterfactual_data_utils import create_single_source_counterfactual_dataset
from utils.das_utils import CNNConfig, CustomLowRankRotatedSpaceIntervention, das_evaluate, das_train
from utils.model_utils import PyTorchCNN
from utils.constants import DATA_DIR, MODEL_DIR, DAS_RESULTS_DIR, NUM_MODELS, LAYERS, VARIABLE_PARTITIONS
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
type_to_dimension_mapping[PyTorchCNN] = []

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
    device : str = 'cuda',
    # display settings
    display_bar : bool = True
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
        # bit of a hack - overwrite latent dim for downstream rotated intervention
        "latent_dim": LAYERS[layer] 
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
        device=device,
        display_bar=display_bar
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
        device=device,
        display_bar=display_bar
    )
    return accuracy

def main(
    coefficients : str,
    image_path : str = 'train_images_2024-12-06_09-18-31.npy',
    label_path : str = 'train_latent_values_2024-12-06_09-18-31.npy',
    n_train: int = 10000,
    n_eval: int = 1000,
    intervention_size: int = 4,
    lr: float = 0.0005,
    num_epochs: int = 3,
    batch_size: int = 512,
    device: str = 'cuda'
):
    images = np.load(f'{DATA_DIR}/{image_path}')
    labels = np.load(f'{DATA_DIR}/{label_path}')
    coef_str = coefficients.split('_')
    np_coefficients = np.array(coef_str, dtype=float) / 100.0

    print('Loaded coefficients:', np_coefficients)

    model_dir = f'{MODEL_DIR}/pytorch_models_ci{coef_str[0]}_co{coef_str[1]}_ar{coef_str[2]}'

    assert len(os.listdir(model_dir)) == NUM_MODELS, f"Expected {NUM_MODELS} in {model_dir}, found {len(os.listdir(model_dir))}"

    results = []
    length = NUM_MODELS * len(VARIABLE_PARTITIONS) * len(LAYERS)
    for model_index, variables, layer in tqdm(itertools.product(range(NUM_MODELS), VARIABLE_PARTITIONS, LAYERS), total=length, desc='Experiments'):
        # load model
        model = PyTorchCNN()
        model.load_state_dict(torch.load(f'{model_dir}/model_{model_index}.pt'))

        print('RUNNING EXPERIMENT:', model_index + 1, variables, layer)

        # run experiment
        accuracy = das_experiment(
            model=model,
            variables=variables,
            layer=layer,
            images=images,
            labels=labels,
            coefficients=np_coefficients,
            n_train=n_train,
            n_eval=n_eval,
            intervention_size=intervention_size,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        results.append({
            'model': model_index,
            'variables': variables,
            'layer': layer,
            'accuracy': accuracy
        })
    results_df = pd.DataFrame(results)

    results_dir = f'{DAS_RESULTS_DIR}/ci{coef_str[0]}_co{coef_str[1]}_ar{coef_str[2]}'
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(f'{results_dir}/results_i{intervention_size}.csv')

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficients', type=str, required=True)
    parser.add_argument('--image_path', type=str, default='train_images_2024-12-06_09-18-31.npy')
    parser.add_argument('--label_path', type=str, default='train_latent_values_2024-12-06_09-18-31.npy')
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_eval', type=int, default=1000)
    parser.add_argument('--intervention_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # run main
    main(
        coefficients=args.coefficients,
        image_path=args.image_path,
        label_path=args.label_path,
        n_train=args.n_train,
        n_eval=args.n_eval,
        intervention_size=args.intervention_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device
    )