import os
import gc
import argparse
from typing import List
from sklearn.model_selection import train_test_split
from utils.model_utils import PyTorchCNN
from utils.probe_utils import probe_train, probe_evaluate, get_activations
from utils.constants import DATA_DIR, LAYERS, MODEL_DIR, NUM_MODELS, PROBE_RESULTS_DIR, VARIABLE_PARTITIONS
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange
from pyvene import type_to_dimension_mapping

random.seed(42)
np.random.seed(42)
_ = torch.manual_seed(42)
_ = torch.cuda.manual_seed(42)

# hack to add custom model to pyvene config
type_to_dimension_mapping[PyTorchCNN] = []

def main(
    coefficients : str,
    image_path : str = 'train_images_2024-12-06_09-18-31.npy',
    label_path : str = 'train_latent_values_2024-12-06_09-18-31.npy',
    n_eval: float = 0.2,
    epochs: int = 1000,
    early_stopping: bool = True,
    patience: int = 20,
    device: str = 'cuda'
):
    # load data
    images = np.load(f'{DATA_DIR}/{image_path}')
    labels = np.load(f'{DATA_DIR}/{label_path}')
    coef_str = coefficients.split('_')
    np_coefficients = np.array(coef_str, dtype=float) / 100.0
    
    model_dir = f'{MODEL_DIR}/pytorch_models_ci{coef_str[0]}_co{coef_str[1]}_ar{coef_str[2]}'
    assert len(os.listdir(model_dir)) == NUM_MODELS, f"Expected {NUM_MODELS} in {model_dir}, found {len(os.listdir(model_dir))}"

    
    X = torch.tensor(images.reshape((-1, 1, 28, 28))).float()
    X_train, X_eval, labels_train, labels_eval = train_test_split(X, labels, test_size=n_eval, random_state=42)

    results = []
    with trange(NUM_MODELS, desc='Experiments') as t:
        for model_index in t:
            # load model
            model = PyTorchCNN()
            model.load_state_dict(torch.load(f'{model_dir}/model_{model_index}.pt'))

            for layer in LAYERS:
                X_act_train = get_activations(model, X_train, layer, device=device)
                X_act_eval = get_activations(model, X_eval, layer, device=device)

                for variables in VARIABLE_PARTITIONS:
                    y_train = torch.tensor(labels_train[:, variables].sum(axis=1)).float()
                    y_eval = torch.tensor(labels_eval[:, variables].sum(axis=1)).float()

                    if layer == 'fc1':
                        epochs = 10000

                    # train probe
                    probe = probe_train(X_act_train, y_train, epochs=epochs, early_stopping=early_stopping, patience=patience, device=device)
                    score = probe_evaluate(probe, X_act_eval, y_eval, device=device)

                    results.append({
                        'model': model_index,
                        'variables': variables,
                        'layer': layer,
                        'score': score
                    })

                    del probe
                    gc.collect()
                    torch.cuda.empty_cache()
                
                del X_act_train, X_act_eval
                gc.collect()
                torch.cuda.empty_cache()

                # save results after each layer (in case of crash)
                results_df = pd.DataFrame(results)
                os.makedirs(PROBE_RESULTS_DIR, exist_ok=True)
                results_path = f'{PROBE_RESULTS_DIR}/ci{coef_str[0]}_co{coef_str[1]}_ar{coef_str[2]}.csv'
                results_df.to_csv(results_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=1)
    parser.add_argument('--n_eval', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(
        model_indices=list(range(args.start_index, args.end_index)),
        n_eval=args.n_eval,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        patience=args.patience,
        device=args.device
    )