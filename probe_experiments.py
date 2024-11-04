import os
import gc
import argparse
from typing import List
from sklearn.model_selection import train_test_split
from utils.model_utils import PyTorchCNN
from utils.probe_utils import probe_train, probe_evaluate, get_activations
from utils.constants import LAYERS, VARIABLE_PARTITIONS
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pyvene import type_to_dimension_mapping

random.seed(42)
np.random.seed(42)
_ = torch.manual_seed(42)
_ = torch.cuda.manual_seed(42)

# hack to add custom model to pyvene config
type_to_dimension_mapping[PyTorchCNN] = []

def main(
    model_indices: List[int],
    n_eval: float = 0.2,
    epochs: int = 1000,
    early_stopping: bool = True,
    patience: int = 20,
    device: str = 'cuda'
):
    # load data
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')
    X = torch.tensor(images.reshape((-1, 1, 28, 28))).float()
    X_train, X_eval, labels_train, labels_eval = train_test_split(X, labels, test_size=n_eval, random_state=42)

    results = []
    with tqdm(model_indices, desc='Experiments') as t:
        for model_index in t:
            # load model
            model = PyTorchCNN()
            model.load_state_dict(torch.load(f'pytorch_models/cnn_model_{model_index + 1}.pth'))

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
                start_index = model_indices[0]
                end_index = model_indices[-1] + 1
                results_dir = 'probe_results'
                os.makedirs(results_dir, exist_ok=True)
                results_df.to_csv(f'{results_dir}/{start_index}_{end_index}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=1)
    parser.add_argument('--n_eval', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=500)
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