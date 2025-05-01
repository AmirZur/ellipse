import argparse
import json
import torch
import torch.nn as nn
import pyvene as pv
from sklearn.metrics import mean_squared_error
from pyvene import type_to_dimension_mapping, LowRankRotatedSpaceIntervention
from tqdm import tqdm, trange
from utils.das_utils import CNNConfig
from utils.counterfactual_data_utils import create_arithmetic_counterfactual_dataset

LAYERS = {
    'layer1': 8,
    'layer2': 4,
    'layer3': 8
}

INTERVENTION_SIZES = {
    'layer1': [1, 2, 4],
    'layer2': [1, 2],
    'layer3': [1, 2, 4]
}

def num_params(model):
    return sum(p.numel() for p in model.parameters())

class PytorchMLP(nn.Module):
    def __init__(self):
        super(PytorchMLP, self).__init__()
        self.layer1 = nn.Linear(2, 8, bias=True)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(8, 4, bias=True)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(4, 8, bias=True)
        self.relu3 = nn.ReLU()
        self.dense = nn.Linear(8, 1, bias=True)
        self.device = self.layer1.weight.device
        self.to(self.device)
    
    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.dense(x)
        return x
    
    def summary(self):
        print("Model Summary:")
        print(f"Layer 1: Linear 2 -> 8 ({num_params(self.layer1)})")
        print(f"Layer 2: Linear 8 -> 4 ({num_params(self.layer2)})")
        print(f"Layer 3: Linear 4 -> 8 ({num_params(self.layer3)})")
        print(f"Dense: Linear 8 -> 1 ({num_params(self.dense)})")


class CustomLowRankRotatedSpaceIntervention(LowRankRotatedSpaceIntervention):
    def __init__(self, **kwargs):
        kwargs["embed_dim"] = kwargs["latent_dim"]
        super().__init__(**kwargs)


type_to_dimension_mapping[PytorchMLP] = []


def interchange_interventions(
    model: PytorchMLP,
    data_size : int = 1000,
    seed : int = 42,
    device : str = 'cuda'
):
    results = []
    for layer in LAYERS:
        model.config = CNNConfig(
            hidden_size=LAYERS[layer],
        )

        representations = [{
            "component": f"{layer}.output",
        }]

        pv_config = pv.IntervenableConfig(
            representations=representations,
            intervention_types=pv.VanillaIntervention
        )
        pv_model = pv.IntervenableModel(pv_config, model)
        pv_model.set_device(device)

        model.device = device

        X_base, X_source, _, _, y_counterfactual_x, y_counterfactual_y = create_arithmetic_counterfactual_dataset(
            size=data_size, seed=seed
        )
        y_counterfactual_x = y_counterfactual_x.cpu().numpy()
        y_counterfactual_y = y_counterfactual_y.cpu().numpy()

        for subspace in range(LAYERS[layer]):
            with torch.no_grad():
                _, preds = pv_model(
                    {'x': X_base.to(device)}, # base
                    [{'x': X_source.to(device)}], # sources (one for each variable)
                    subspaces=subspace # only intervene on i-th neuron
                )
            
            preds = preds.squeeze().cpu().numpy()

            mse_x = mean_squared_error(preds, y_counterfactual_x)
            mse_y = mean_squared_error(preds, y_counterfactual_y)
            results.append({
                "layer": layer,
                "subspace": subspace,
                "variable": "x",
                "mse": mse_x,
                "intervention_type": "interchange"
            })
            results.append({
                "layer": layer,
                "subspace": subspace,
                "variable": "y",
                "mse": mse_y,
                "intervention_type": "interchange"
            })
    return results

def das_train(
    pv_model, X_base, X_source, y_counterfactual, 
    lr=0.0001, num_epochs=5, batch_size=1000, subspaces=None, device='cuda', display_bar=True
):
    pv_model.train()
    optimizer = torch.optim.Adam(pv_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        with trange(0, X_base.shape[0], batch_size, desc=f'Training (Epoch {epoch+1})', disable=not display_bar) as progress_bar:
            for b in progress_bar:
                optimizer.zero_grad()
                _, preds = pv_model(
                    {'x': X_base[b:b+batch_size].to(device)}, # base
                    [{'x': X_source[b:b+batch_size].to(device)}], # sources (one for each variable)
                    subspaces=subspaces # intervene on first intervention_size neurons
                )
                loss_fn = nn.MSELoss()
                loss = loss_fn(preds, y_counterfactual[b:b+batch_size].view(-1, 1).to(device))
                progress_bar.set_postfix({'loss': loss.item()})
                loss.backward()
                optimizer.step()

def das(
    model: PytorchMLP,
    train_size : int = 100000,
    eval_size : int = 1000,
    train_seed : int = 0,
    eval_seed : int = 42,
    device : str = 'cuda'
):
    X_base_train, X_source_train, _, _, y_counterfactual_x_train, y_counterfactual_y_train = create_arithmetic_counterfactual_dataset(
        size=train_size, seed=train_seed
    )
    X_base_eval, X_source_eval, _, _, y_counterfactual_x_eval, y_counterfactual_y_eval = create_arithmetic_counterfactual_dataset(
        size=eval_size, seed=eval_seed
    )
    results = []
    for layer in LAYERS:
        model.config = CNNConfig(
            hidden_size=LAYERS[layer],
        )

        for intervention_size in INTERVENTION_SIZES[layer]:

            for variable in ["x", "y"]:
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

                model.device = device

                das_train(
                    pv_model, X_base_train, X_source_train, y_counterfactual_x_train if variable == "x" else y_counterfactual_y_train,
                    lr=0.01, num_epochs=50, batch_size=10000, display_bar=False
                )

                with torch.no_grad():
                    _, preds = pv_model(
                        {'x': X_base_eval.to(device)}, # base
                        [{'x': X_source_eval.to(device)}], # sources (one for each variable)
                        subspaces=None
                    )
                
                preds = preds.squeeze().cpu().numpy()
                mse = mean_squared_error(preds, y_counterfactual_x_eval if variable == "x" else y_counterfactual_y_eval)
                results.append({
                    "layer": layer,
                    "intervention_size": intervention_size,
                    "variable": variable,
                    "mse": mse,
                    "intervention_type": "das"
                })
    return results


def main(
    saved_models : str = "data/arithmetic/pytorch_models.pt"
):
    models = torch.load(saved_models, weights_only=False)
    all_results = []
    for model in tqdm(models, desc="Analyzing models..."):
        results_interchange = interchange_interventions(
            model,
        )
        results_das = das(
            model,
        )
        results = results_interchange + results_das
        for r in results:
            r["model_index"] = len(all_results)

        all_results.append(results)
    
    with open('arithmetic_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to arithmetic_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arithmetic causal analysis")
    parser.add_argument("--saved_models", type=str, default="data/arithmetic/pytorch_models.pt", help="Path to saved models")
    args = parser.parse_args()
    main(args.saved_models)