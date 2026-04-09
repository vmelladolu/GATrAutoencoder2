"""
Hungarian matching loss for set-to-set comparison.

Uses scipy.optimize.linear_sum_assignment for optimal bipartite matching
(non-differentiable), then computes MSE on matched pairs (differentiable).
This is the standard DETR-style approach.

Cost matrix is built over all hit features: (x, y, z, k, thr).
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_loss_batch(
    gen_xyz,    # (B, n_gen, 3)
    gen_k,      # (B, n_gen, 1)
    gen_thr,    # (B, n_gen, 1)
    true_xyz,   # (N_real_total, 3) — flat, all events concatenated
    true_k,     # (N_real_total, 1)
    true_thr,   # (N_real_total, 1)
    batch_idx,  # (N_real_total,) — event index per real hit
    w_xyz=1.0,
    w_k=1.0,
    w_thr=1.0,
):
    """
    Hungarian matching loss over a batch.

    For each event:
      1. Build cost matrix (n_real × n_gen) via L2 over all features.
      2. Solve assignment: each real hit is matched to a unique generated hit.
         n_gen >= n_real is guaranteed when caller uses ceil(max_N / d) steps.
      3. MSE on matched pairs — differentiable via straight-through.

    Unassigned generated hits (if n_gen > n_real) contribute nothing to the loss.

    Returns:
        Scalar mean loss over all events in the batch.
    """
    B      = gen_xyz.shape[0]
    device = gen_xyz.device
    losses = []

    for b in range(B):
        mask  = (batch_idx == b)
        t_xyz = true_xyz[mask]    # (n_real, 3)
        t_k   = true_k[mask]      # (n_real, 1)
        t_thr = true_thr[mask]    # (n_real, 1)
        n_real = t_xyz.shape[0]

        if n_real == 0:
            # Empty event: contribute zero loss without breaking the graph
            losses.append(gen_xyz[b].sum() * 0.0)
            continue

        p_xyz = gen_xyz[b]    # (n_gen, 3)
        p_k   = gen_k[b]      # (n_gen, 1)
        p_thr = gen_thr[b]    # (n_gen, 1)

        # --- Build cost matrix (non-differentiable) ---
        with torch.no_grad():
            pred_feat = torch.cat([p_xyz, p_k, p_thr], dim=-1).float()   # (n_gen, 5)
            true_feat = torch.cat([t_xyz, t_k, t_thr], dim=-1).float()   # (n_real, 5)
            # cost[i, j] = ||true_feat[i] - pred_feat[j]||^2
            cost = torch.cdist(true_feat, pred_feat, p=2).pow(2)          # (n_real, n_gen)
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

        # --- MSE on matched pairs (differentiable) ---
        loss_xyz = F.mse_loss(p_xyz[col_ind], t_xyz[row_ind])
        loss_k   = F.mse_loss(p_k[col_ind],   t_k[row_ind])
        loss_thr = F.mse_loss(p_thr[col_ind], t_thr[row_ind])

        losses.append(w_xyz * loss_xyz + w_k * loss_k + w_thr * loss_thr)

    return torch.stack(losses).mean()
