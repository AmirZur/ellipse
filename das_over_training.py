import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from pyvene import type_to_dimension_mapping
from utils.model_utils import train, evaluate, PyTorchCNN
from das_experiments import das_experiment

type_to_dimension_mapping[PyTorchCNN] = []

random.seed(42)
np.random.seed(42)
_ = torch.manual_seed(42)
_ = torch.cuda.manual_seed(42)

COEFFICIENTS = [
    np.array([0.4, 0.4, 0.4]),
    np.array([0.3, 0.4, 0.5]),
    np.array([0.5, 0.4, 0.3])
]

def copy_model(model):
    model_copy = PyTorchCNN().to('cuda')
    model_copy.load_state_dict(model.state_dict())
    return model_copy

def main(
    output_dir: str = 'das_over_training',
    coef_index: int = 0,
    intervention_size : int = 4,
    batch_size : int = 1024,
    num_epochs : int = 10
):
    # create model
    model = PyTorchCNN().to('cuda')

    # load data
    labels = np.load('data/labels.npy')
    images = np.load('data/images.npy')
    coefficients = COEFFICIENTS[coef_index]

    # batch, channel, height, width
    X_train = torch.tensor(images.reshape((-1, 1, 28, 28))).float().to('cuda')
    y_train = torch.tensor(np.matmul(labels, coefficients) > 0.6).float().to('cuda')

    data = {0: [], 1: [], 2: []}

    def callback(model, variables):
        accuracy = das_experiment(
            copy_model(model), variables, 'conv3', images, labels, coefficients, 
            device='cuda', display_bar=False, intervention_size=intervention_size
        )
        data[variables[0]].append(accuracy)

    callbacks = [
        lambda m: callback(m, [0]),
        lambda m: callback(m, [1]),
        lambda m: callback(m, [2])
    ]

    train(model, X_train, y_train, batch_size=batch_size, num_epochs=num_epochs, callback_fns=callbacks)

    train_accuracy = evaluate(model, X_train, y_train, batch_size=batch_size)
    print(f'Train accuracy: {train_accuracy:.4f}')

    df = pd.DataFrame(data)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/results_i{intervention_size}_c{coef_index}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='das_over_training')
    parser.add_argument('--coef_index', type=int, default=0)
    parser.add_argument('--intervention_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()
    main(**vars(args))
