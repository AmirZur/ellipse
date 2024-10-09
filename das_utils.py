from tqdm import trange
import torch
import torch.nn as nn
from model_utils import PyTorchCNN
from pyvene import type_to_dimension_mapping

type_to_dimension_mapping[PyTorchCNN] = []

class CNNConfig:
    def __init__(self, hidden_size=448):
        self.hidden_size = hidden_size

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size
        }

def das_train(
    pv_model, X_base, X_sources, y_counterfactual, 
    lr=0.0001, num_epochs=1, batch_size=256, subspaces=0
):
    pv_model.train()
    optimizer = torch.optim.Adam(pv_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        with trange(0, X_base.shape[0], batch_size, desc=f'Training (Epoch {epoch+1})') as progress_bar:
            for b in progress_bar:
                optimizer.zero_grad()
                _, preds = pv_model(
                    {'x': X_base[b:b+batch_size]}, # base
                    [{'x': X_sources[i][b:b+batch_size]} for i in range(X_sources.shape[0])], # sources (one for each variable)
                    subspaces=subspaces # intervene on first intervention_size neurons
                )
                loss_fn = nn.BCELoss()
                loss = loss_fn(preds, y_counterfactual[b:b+batch_size].view(-1, 1))
                progress_bar.set_postfix({'loss': loss.item()})
                loss.backward()
                optimizer.step()

def das_predict(pv_model, X_base, X_sources, batch_size=256):
    pv_model.eval()

    y_pred = None
    for b in trange(0, X_base.shape[0], batch_size, desc='Evaluating'):
        with torch.no_grad():
            _, y_pred_batch = pv_model(
                {'x': X_base[b:b+batch_size]}, # base
                [{'x': X_sources[i][b:b+batch_size]} for i in range(X_sources.shape[0])], # sources
                subspaces=0 # intervene on first intervention_size neurons
            )
        y_pred_batch = y_pred_batch.round().flatten()
        if y_pred is None:
            y_pred = y_pred_batch
        else:
            y_pred = torch.cat((y_pred, y_pred_batch))
    return y_pred

def das_evaluate(pv_model, X_base, X_sources, y_counterfactual, batch_size=256):
    y_pred = das_predict(pv_model, X_base, X_sources, batch_size=batch_size)
    accuracy = (y_pred == y_counterfactual).float().mean()
    return accuracy.item()

def iit_train(
    pv_model, X_base, X_sources, y_counterfactual, 
    lr=0.0001, num_epochs=1, batch_size=256, subspaces=0
):
    pv_model.enable_model_gradients()
    das_train(pv_model, X_base, X_sources, y_counterfactual, lr, num_epochs, batch_size, subspaces)

iit_predict = das_predict
iit_evaluate = das_evaluate