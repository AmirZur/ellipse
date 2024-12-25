import numpy as np
import torch

def create_single_source_counterfactual_dataset(variables, images, labels, coefficients, size=10000, return_labels=False):
    """
    Create counterfactual dataset for a given set of high level variables.
    Uses a SINGLE source for ALL variables (i.e., supports an abstraction where the variables are combined).
    """
    # randomly sample base images/labels
    base_indices = np.random.choice(np.arange(images.shape[0]), size=size, replace=True)
    # sample a single random source image/label for all variables
    source_indices = np.random.choice(np.arange(images.shape[0]), size=size, replace=True)

    base_images = images[base_indices] # list of base images
    source_images = images[source_indices] # list of source images (one for all variables)
    base_labels = labels[base_indices]
    source_labels = labels[source_indices]

    X_base = torch.tensor(base_images.reshape((-1, 1, 28, 28))).float()
    X_sources = torch.tensor(source_images.reshape((-1, 1, 28, 28))).float()
    # labels for base/source examples WITHOUT intervention
    y_base = torch.tensor(np.matmul(base_labels, coefficients) > 0.6).float()
    y_sources = torch.tensor(np.matmul(source_labels, coefficients) > 0.6).float()

    # to create the counterfactual label (WITH intervention matching all variables -> single source),
    # take the variables from the source, the rest from the base
    coefficients_source = np.zeros_like(coefficients)
    # restore coefficient for each variable we're intervening on
    for variable in variables:
        coefficients_source[variable] = coefficients[variable]

    # for base, zero out coefficients for variables we're intervening on
    coefficients_base = coefficients.copy()
    coefficients_base[variables] = 0.

    y_counterfactual = np.matmul(base_labels, coefficients_base) + np.matmul(source_labels, coefficients_source)
    y_counterfactual = torch.tensor(y_counterfactual > 0.6).float()

    # to standardize the return values, we wrap the single source in a list
    X_sources = X_sources.unsqueeze(0)
    y_sources = y_sources.unsqueeze(0)
    source_labels = [source_labels]

    if return_labels:
        return X_base, X_sources, y_base, y_sources, y_counterfactual, base_labels, source_labels

    return X_base, X_sources, y_base, y_sources, y_counterfactual

# def create_multi_source_counterfactual_dataset(variables, images, labels, coefficients, size=10000, return_labels=False):
#     """
#     Create counterfactual dataset for a given set of high level variables.
#     Uses a SEPARATE source FOR EACH variable.
#     """
#     # randomly sample base images/labels
#     base_indices = np.random.choice(np.arange(images.shape[0]), size=size, replace=True)
#     # for each variable, sample a random source image/label
#     source_indices = [
#         np.random.choice(np.arange(images.shape[0]), size=size, replace=True)
#         for _ in variables
#     ]

#     base_images = images[base_indices] # list of base images
#     source_images = [images[src_idx] for src_idx in source_indices] # list of lists of source images (one list per variable)
#     base_labels = labels[base_indices]
#     source_labels = [labels[src_idx] for src_idx in source_indices]

#     X_base = torch.tensor(base_images.reshape((-1, 1, 28, 28))).float()
#     X_sources = torch.stack([torch.tensor(src_img.reshape((-1, 1, 28, 28))).float() for src_img in source_images])
#     # labels for base/source examples WITHOUT intervention
#     y_base = torch.tensor(np.matmul(base_labels, coefficients) > 0.6).float()
#     y_sources = torch.stack([torch.tensor(np.matmul(src_lbl, coefficients) > 0.6).float() for src_lbl in source_labels])

#     # to create the counterfactual label (WITH intervention matching source i -> variable i),
#     # take i-th variable from source i, the rest from base
#     coefficients_sources = [np.zeros_like(coefficients) for _ in variables]
#     # restore coefficient for each variable we're intervening on
#     for i, variable in enumerate(variables):
#         coefficients_sources[i][variable] = coefficients[variable]

#     # for base, zero out coefficients for variables we're intervening on
#     coefficients_base = coefficients.copy()
#     coefficients_base[variables] = 0.

#     y_counterfactual = np.matmul(base_labels, coefficients_base) + np.sum([
#         np.matmul(source_labels[i], coefficients_sources[i])
#         for i, _ in enumerate(variables)
#     ], axis=0)
#     y_counterfactual = torch.tensor(y_counterfactual > 0.6).float()

#     if return_labels:
#         return X_base, X_sources, y_base, y_sources, y_counterfactual, base_labels, source_labels

#     return X_base, X_sources, y_base, y_sources, y_counterfactual