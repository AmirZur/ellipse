from tqdm import trange
import torch
import torch.nn as nn
from pyvene import LowRankRotatedSpaceIntervention

##############################################
# pyvene helpers (a bit hacky)               #
##############################################
class CNNConfig:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size
        }

class CustomLowRankRotatedSpaceIntervention(LowRankRotatedSpaceIntervention):
    def forward(self, base, source, subspaces=None):
        # reshape: (batch x everything else)
        base_flattened = base.view(base.shape[0], -1)
        source_flattened = source.view(source.shape[0], -1)
        output = super().forward(base_flattened, source_flattened, subspaces)
        return output.view(base.shape)

    
##############################################
# DAS helper methods                         #
##############################################

def das_train(
    pv_model, X_base, X_sources, y_counterfactual, 
    lr=0.0001, num_epochs=1, batch_size=256, subspaces=None, device='cuda'
):
    pv_model.train()
    optimizer = torch.optim.Adam(pv_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        with trange(0, X_base.shape[0], batch_size, desc=f'Training (Epoch {epoch+1})') as progress_bar:
            for b in progress_bar:
                optimizer.zero_grad()
                _, preds = pv_model(
                    {'x': X_base[b:b+batch_size].to(device)}, # base
                    [{'x': X_sources[i][b:b+batch_size].to(device)} for i in range(X_sources.shape[0])], # sources (one for each variable)
                    subspaces=subspaces # intervene on first intervention_size neurons
                )
                loss_fn = nn.BCELoss()
                loss = loss_fn(preds, y_counterfactual[b:b+batch_size].view(-1, 1).to(device))
                progress_bar.set_postfix({'loss': loss.item()})
                loss.backward()
                optimizer.step()

def das_predict(pv_model, X_base, X_sources, batch_size=256, subspaces=None, device='cuda'):
    pv_model.eval()

    y_pred = None
    for b in trange(0, X_base.shape[0], batch_size, desc='Evaluating'):
        with torch.no_grad():
            _, y_pred_batch = pv_model(
                {'x': X_base[b:b+batch_size].to(device)}, # base
                [{'x': X_sources[i][b:b+batch_size].to(device)} for i in range(X_sources.shape[0])], # sources
                subspaces=subspaces # intervene on first intervention_size neurons
            )
        y_pred_batch = y_pred_batch.round().flatten().cpu()
        if y_pred is None:
            y_pred = y_pred_batch
        else:
            y_pred = torch.cat((y_pred, y_pred_batch))
    return y_pred

def das_evaluate(pv_model, X_base, X_sources, y_counterfactual, batch_size=256, subspaces=None, device='cuda'):
    y_pred = das_predict(pv_model, X_base, X_sources, batch_size=batch_size, subspaces=subspaces, device=device)
    accuracy = (y_pred == y_counterfactual).float().mean()
    return accuracy.item()

##############################################
# IIT helper methods (wrap DAS methods)      #
##############################################

def iit_train(
    pv_model, X_base, X_sources, y_counterfactual, 
    lr=0.0001, num_epochs=1, batch_size=256, subspaces=0
):
    pv_model.enable_model_gradients()
    das_train(pv_model, X_base, X_sources, y_counterfactual, lr, num_epochs, batch_size, subspaces)

iit_predict = das_predict
iit_evaluate = das_evaluate