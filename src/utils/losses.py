import numpy as np
import torch


def reconstruction_loss(outputs, logenergy, class_weights=None):
    logenergy = logenergy.view(-1)
    if class_weights is not None:
        weights = torch.ones_like(logenergy)
        for e_str, w in class_weights.items():
            e_val = float(e_str)
            mask = (logenergy >= np.log(e_val - 0.5)) & (logenergy < np.log(e_val + 0.5))
            weights[mask] = w
        return torch.mean(weights * (outputs.squeeze() - logenergy) ** 2)
    return torch.mean((outputs.squeeze() - logenergy) ** 2)

