import builtins
import os
import wandb
import time
import torch



def wprint(*args, **kwargs):
    builtins.print(*args, **kwargs)
    if wandb is None or wandb.run is None:
        return
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args) + end
    wandb.termlog(message.rstrip("\n"))


def _log_gradient_stats(model, step=None):
    if wandb is None:
        return
    total_norm = 0.0
    grad_log = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total_norm += param_norm ** 2
        grad_log[f"grads/{name.replace('.', '/')}"] = param_norm
    total_norm = total_norm ** 0.5
    grad_log["grads/total_norm"] = total_norm
    wandb.log(grad_log)


def _log_prediction_debug(outputs, target, step=None):
    if wandb is None:
        return
    pred = outputs.detach().cpu()
    tgt = target.detach().cpu().view(-1)
    corr = float("nan")
    if pred.numel() > 1:
        corr = float(torch.corrcoef(torch.stack([pred.view(-1), tgt]))[0, 1])
    wandb.log({
        "debug/pred_mean": pred.mean().item(),
        "debug/pred_std": pred.std().item(),
        "debug/pred_min": pred.min().item(),
        "debug/pred_max": pred.max().item(),
        "debug/target_mean": tgt.mean().item(),
        "debug/target_std": tgt.std().item(),
        "debug/target_min": tgt.min().item(),
        "debug/target_max": tgt.max().item(),
        "debug/pred_target_corr": corr,
    })


def _log_aggregation_debug(model, mv_v_part, mv_s_part, scalars, batch_idx, step=None):
    if wandb is None:
        return
    with torch.no_grad():
        from torch_scatter import scatter_mean as _sm

        mv_latent, s_latent, _, _ = model.encode(mv_v_part, mv_s_part, scalars, batch_idx)
        mv_agg = _sm(mv_latent.squeeze(1), batch_idx, dim=0)
        s_agg = _sm(s_latent, batch_idx, dim=0)
        agg = torch.cat([mv_agg, s_agg], dim=-1)
        std_per_dim = agg.std(dim=0)
        wandb.log({
            "debug/latent_agg_mean": agg.mean().item(),
            "debug/latent_agg_std": agg.std().item(),
            "debug/latent_agg_max": agg.max().item(),
            "debug/latent_agg_min": agg.min().item(),
            "debug/latent_mv_std": mv_agg.std().item(),
            "debug/latent_s_std": s_agg.std().item(),
            "debug/latent_agg_dim_std_mean": std_per_dim.mean().item(),
            "debug/latent_dead_dims": int((std_per_dim < 1e-5).sum().item()),
        })
