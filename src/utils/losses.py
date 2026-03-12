import numpy as np
import torch
import torch.nn.functional as F


def reconstruction_loss(outputs, logenergy, class_weights=None, loss_type="mse", huber_delta=1.0):
    logenergy = logenergy.view(-1)
    diff = outputs.squeeze() - logenergy

    if loss_type == "mse":
        per_sample_loss = diff ** 2
    elif loss_type == "huber":
        per_sample_loss = F.smooth_l1_loss(outputs.squeeze(), logenergy, beta=huber_delta, reduction='none')
    elif loss_type == "log_cosh":
        per_sample_loss = torch.log(torch.cosh(diff + 1e-12))
    else:
        raise ValueError(f"Loss type no soportado: {loss_type}")

    if class_weights is not None:
        meta = class_weights.get("__meta__", {})
        half_w = meta.get("bin_half_width", 0.5)

        weights = torch.ones_like(logenergy)
        for e_str, w in class_weights.items():
            if e_str == "__meta__":
                continue
            e_val = float(e_str)
            lo = np.log(max(e_val - half_w, 1e-6))
            hi = np.log(e_val + half_w)
            mask = (logenergy >= lo) & (logenergy < hi)
            weights[mask] = w
        return torch.mean(weights * per_sample_loss)

    return torch.mean(per_sample_loss)

