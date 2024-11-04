import torch
from utils.model_utils import PyTorchCNN
from utils.das_utils import CNNConfig
from utils.constants import LAYERS
import pyvene as pv
from tqdm import trange

class LinearProbe(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, X, y, epochs=100, device='cuda', early_stopping=True, patience=10):
        self.to(device)
        X = X.to(device)
        y = y.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        best_loss = float('inf')
        best_loss_epoch = 0
        with trange(epochs) as t:
            for epoch in t:
                optimizer.zero_grad()
                y_pred = self(X)
                loss = criterion(y_pred, y)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_loss_epoch = epoch
                if early_stopping and epoch - best_loss_epoch > patience:
                    print(f'Early stopping at epoch {epoch} (best loss: {best_loss})')
                    break

                loss.backward()
                t.set_postfix(loss=loss.item())
                optimizer.step()
    
    def score(self, X, y, device='cuda'):
        self.to(device)
        X = X.to(device)
        y = y.to(device)
        y_pred = self(X)
        # compute R^2
        R_2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return R_2.item()

def get_activations(model : PyTorchCNN, X : torch.FloatTensor, layer : str, device='cuda') -> torch.FloatTensor:
    model.eval()
    model.config = CNNConfig(
        hidden_size=LAYERS[layer],
    )

    # config to save layer outputs
    representations = [{
        "component": f"{layer}.output",
    }]

    pv_config = pv.IntervenableConfig(
        representations=representations,
        intervention_types=pv.CollectIntervention
    )
    pv_model = pv.IntervenableModel(pv_config, model)
    pv_model.set_device(device)

    (_, activations), _ = pv_model(base={'x': X.to(device)})
    return torch.stack(activations).flatten(1)


def probe_train(X : torch.FloatTensor, y : torch.FloatTensor, epochs=500, early_stopping=True, patience=20, device='cuda') -> LinearProbe:
    if y.ndim == 1:
        y = y.unsqueeze(1)
    probe = LinearProbe(X.shape[1], 1)
    probe.fit(X, y, epochs=epochs, early_stopping=early_stopping, patience=patience, device=device)
    return probe

def probe_evaluate(probe : LinearProbe, X : torch.FloatTensor, y : torch.FloatTensor, device='cuda') -> float:
    if y.ndim == 1:
        y = y.unsqueeze(1)
    return probe.score(X, y, device=device)