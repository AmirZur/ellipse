from tqdm import trange
import torch
import torch.nn as nn
from pyvene import LowRankRotatedSpaceIntervention
from pyvene.models.layers import LowRankRotateLayer

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
    def __init__(self, **kwargs):
        kwargs["embed_dim"] = kwargs["latent_dim"]
        super().__init__(**kwargs)

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
    lr=0.0001, num_epochs=1, batch_size=256, subspaces=None, device='cuda', display_bar=True
):
    pv_model.train()
    optimizer = torch.optim.Adam(pv_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        with trange(0, X_base.shape[0], batch_size, desc=f'Training (Epoch {epoch+1})', disable=not display_bar) as progress_bar:
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

def das_predict(pv_model, X_base, X_sources, batch_size=256, subspaces=None, device='cuda', display_bar=True):
    pv_model.eval()

    y_pred = None
    for b in trange(0, X_base.shape[0], batch_size, desc='Evaluating', disable=not display_bar):
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

def das_evaluate(pv_model, X_base, X_sources, y_counterfactual, batch_size=256, subspaces=None, device='cuda', display_bar=True):
    y_pred = das_predict(pv_model, X_base, X_sources, batch_size=batch_size, subspaces=subspaces, device=device, display_bar=display_bar)
    accuracy = (y_pred == y_counterfactual).float().mean()
    return accuracy.item()

##############################################
# IIT helper methods (wrap DAS methods)      #
##############################################

def iit_train(
    pv_model, X_base, X_sources, y_counterfactual, 
    lr=0.0001, num_epochs=1, batch_size=256, subspaces=None, device='cuda'
):
    pv_model.enable_model_gradients()
    das_train(pv_model, X_base, X_sources, y_counterfactual, lr, num_epochs, batch_size, subspaces, device)

iit_predict = das_predict
iit_evaluate = das_evaluate


##############################################
# Adversarial IIT helper methods             #
##############################################
def toggle_intervention_gradients(pv_model, intervention_key, enable=True):
    for p in pv_model.interventions[intervention_key][0].parameters():
        p.requires_grad_(enable)

def iit_adversarial_attack(
    pv_model,
    X_base_adv, X_sources_adv, y_counterfactual_adv,
    adv_lr=0.0001, num_adv_steps=5, subspaces=None, device='cuda'
):
    optimizer = torch.optim.Adam(pv_model.parameters(), lr=adv_lr)
    for _ in range(num_adv_steps):
        optimizer.zero_grad()
        _, preds = pv_model(
            {'x': X_base_adv.to(device)}, # base
            [None, {'x': X_sources_adv[0].to(device)}], # only intervene with adversarial intervention
            subspaces=subspaces # intervene on first intervention_size neurons
        )
        loss_fn = nn.BCELoss()
        loss = loss_fn(preds, y_counterfactual_adv.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
    # return final (non-backpropagated) loss
    _, preds = pv_model(
        {'x': X_base_adv.to(device)}, # base
        [None, {'x': X_sources_adv[0].to(device)}],# only intervene with adversarial intervention
        subspaces=subspaces # intervene on first intervention_size neurons
    )
    loss_fn = nn.BCELoss()
    loss = loss_fn(preds, y_counterfactual_adv.view(-1, 1).to(device))
    return loss

def cutoff_loss(loss, cutoff):
    cutoff_r = 0.999
    loss_r = 1 - cutoff_r
    # if below, interpolate between cutoff and loss
    below = (loss < cutoff).float() * (cutoff_r * cutoff + loss_r * loss)
    # if above, just return loss
    above = (loss >= cutoff).float() * loss
    return below + above

def iit_adversarial_train(
    pv_model, 
    X_base, X_sources, y_counterfactual,
    X_base_adv, X_sources_adv, y_counterfactual_adv,
    lr=0.0001, num_epochs=1, batch_size=256, subspaces=None, device='cuda',
    adv_lr=0.0001, num_adv_steps=5, cutoff=0.5
):
    # NOTE: assumes that 1st intervention is clean and 2nd intervention is adversarial
    intervention_keys = list(pv_model.interventions.keys())
    clean_intervention_key = intervention_keys[0]
    adv_intervention_key = intervention_keys[1]

    pv_model.train()
    pv_model.enable_model_gradients()
    optimizer = torch.optim.Adam(pv_model.parameters(), lr=lr)

    def toggle_for_adv_attack(pv_model, before_adv=True):
        if before_adv:
            # re-initialize the adversarial intervention (learning from scratch)
            pv_model.interventions[adv_intervention_key] = (
                CustomLowRankRotatedSpaceIntervention(
                    embed_dim=pv_model.interventions[adv_intervention_key][0].embed_dim.item(), 
                    low_rank_dimension=4
                ).to(device),
                pv_model.interventions[adv_intervention_key][1]
            )
            pv_model.disable_model_gradients()
            toggle_intervention_gradients(pv_model, clean_intervention_key, enable=False)
            toggle_intervention_gradients(pv_model, adv_intervention_key, enable=True)
        else:
            pv_model.enable_model_gradients()
            toggle_intervention_gradients(pv_model, clean_intervention_key, enable=True)
            toggle_intervention_gradients(pv_model, adv_intervention_key, enable=False)

    for epoch in range(num_epochs):
        with trange(0, X_base.shape[0], batch_size, desc=f'Training (Epoch {epoch+1})') as progress_bar:
            for b in progress_bar:
                optimizer.zero_grad()

                toggle_for_adv_attack(pv_model, before_adv=True)
                adv_sources = torch.stack([s[b:b+batch_size] for s  in X_sources_adv])
                adv_loss = iit_adversarial_attack(
                    pv_model, X_base_adv[b:b+batch_size], adv_sources, y_counterfactual_adv[b:b+batch_size], 
                    adv_lr=adv_lr, num_adv_steps=num_adv_steps, device=device
                )
                toggle_for_adv_attack(pv_model, before_adv=False)
                
                _, preds = pv_model(
                    {'x': X_base[b:b+batch_size].to(device)}, # base
                    [{'x': X_sources[0][b:b+batch_size].to(device)}, None], # only intervene with clean intervention
                    subspaces=subspaces # intervene on first intervention_size neurons
                )
                loss_fn = nn.BCELoss()
                original_loss = loss_fn(preds, y_counterfactual[b:b+batch_size].view(-1, 1).to(device))
                original_loss = cutoff_loss(original_loss, cutoff)

                loss = original_loss - adv_loss
                progress_bar.set_postfix({'loss': loss.item(), 'clean_loss': original_loss.item(), 'adv_loss': adv_loss.item()})
                loss.backward()
                optimizer.step()

def iit_adversarial_predict(
    pv_model, 
    X_base, 
    X_sources, 
    clean : bool = True, 
    batch_size=256, 
    subspaces=None, 
    device='cuda'
):
    pv_model.eval()

    y_pred = None
    with trange(0, X_base.shape[0], batch_size, desc='Evaluating') as progress_bar:
        for b in progress_bar:
            if clean:
                sources = [{'x': X_sources[0][b:b+batch_size].to(device)}, None]
            else:
                sources = [None, {'x': X_sources[0][b:b+batch_size].to(device)}]

            with torch.no_grad():
                _, y_pred_batch = pv_model(
                    {'x': X_base[b:b+batch_size].to(device)}, # base
                    sources, # sources (one for each variable)
                    subspaces=subspaces # intervene on first intervention_size neurons
                )
            y_pred_batch = y_pred_batch.round().flatten().cpu()
            if y_pred is None:
                y_pred = y_pred_batch
            else:
                y_pred = torch.cat((y_pred, y_pred_batch))
    return y_pred

def iit_adversarial_evaluate(pv_model, X_base, X_sources, y_counterfactual, clean, batch_size=256, subspaces=None, device='cuda'):
    y_pred = iit_adversarial_predict(pv_model, X_base, X_sources, clean, batch_size=batch_size, subspaces=subspaces, device=device)
    accuracy = (y_pred == y_counterfactual).float().mean()
    return accuracy.item()