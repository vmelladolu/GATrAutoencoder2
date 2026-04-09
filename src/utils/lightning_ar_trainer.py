import math
import os
import time

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from src.models.gatr_ar_autoencoder import GATrARAutoencoder
from src.utils.batch_utils import build_batch
from src.utils.hungarian_loss import hungarian_loss_batch
from src.utils.logging import wprint, _log_gradient_stats

try:
    import wandb
except ImportError:
    wandb = None


class LightningGATrARAutoencoder(L.LightningModule):
    """
    Lightning module for GATrARAutoencoder.

    Training is split into two phases:

    Phase 1 (epochs 0 .. pretrain_n_epochs-1):
        - Decoder FROZEN (no gradients).
        - Trains encoder + aggregation + VAE head + head N.
        - Loss: loss_n + beta_kl * loss_kl

    Phase 2 (epochs pretrain_n_epochs .. max_epochs-1):
        - All parameters trainable.
        - Full autoregressive loop + Hungarian matching.
        - Loss: lambda_hungarian * loss_h + lambda_n * loss_n + beta_kl * loss_kl
    """

    def __init__(
        self,
        cfg_enc,
        cfg_dec,
        cfg_agg,
        cfg_n_head,
        cfg_vae,
        pretrain_n_epochs,
        lambda_hungarian,
        lambda_n,
        d_step,
        hungarian_weights,      # dict: w_xyz, w_k, w_thr
        use_scalar,
        use_one_hot,
        use_time,
        learning_rate,
        max_epochs,
        output_path,
        scheduled_sampling_T=0, # epochs to linearly transition n_steps GT→predicted (0 = always GT)
        scheduler_cfg=None,
        optimizer_cfg=None,
        debug_cfg=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])

        use_vae         = cfg_vae.get("use_vae", False)
        event_embed_dim = cfg_vae.get("event_embed_dim", 32)
        self.beta_kl    = cfg_vae.get("beta_kl", 0.0)

        self.model = GATrARAutoencoder(
            cfg_enc=cfg_enc,
            cfg_dec=cfg_dec,
            cfg_agg=cfg_agg,
            cfg_n_head=cfg_n_head,
            d=d_step,
            use_vae=use_vae,
            event_embed_dim=event_embed_dim,
        )

        # Freeze decoder at start; unfrozen at beginning of phase 2.
        self._freeze_decoder()

        self.pretrain_n_epochs    = pretrain_n_epochs
        self.d_step               = d_step
        self.lambda_hungarian     = lambda_hungarian
        self.lambda_n             = lambda_n
        self.scheduled_sampling_T = scheduled_sampling_T
        self.hungarian_w       = hungarian_weights if hungarian_weights else {}
        self.use_scalar        = use_scalar
        self.use_one_hot       = use_one_hot
        self.use_time          = use_time
        self.learning_rate     = learning_rate
        self.max_epochs        = max_epochs
        self.output_path       = output_path
        self.scheduler_cfg     = scheduler_cfg if scheduler_cfg else {"type": "cosine"}
        self.optimizer_cfg     = optimizer_cfg if optimizer_cfg else {}
        self.debug_cfg         = debug_cfg if debug_cfg else {}

        self.use_ema     = self.optimizer_cfg.get("ema", False)
        self.ema_decay   = self.optimizer_cfg.get("ema_decay", 0.999)
        self._ema        = None
        self._ema_state_to_load = None

        self._epoch_start_time = None
        self._batch_start_time = None
        self._debug_grad_step  = None

        self._val_losses_local = []

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _freeze_decoder(self):
        for name in ("decoder", "thr_head", "output_to_input_proj"):
            module = getattr(self.model, name)
            for p in module.parameters():
                p.requires_grad_(False)

    def _unfreeze_decoder(self):
        for name in ("decoder", "thr_head", "output_to_input_proj"):
            module = getattr(self.model, name)
            for p in module.parameters():
                p.requires_grad_(True)

    @property
    def _is_phase2(self):
        return self.current_epoch >= self.pretrain_n_epochs

    @property
    def _ss_alpha(self):
        """Fraction of predicted n_steps to use (0.0 = all GT, 1.0 = all predicted)."""
        if not self._is_phase2 or self.scheduled_sampling_T <= 0:
            return 0.0
        phase2_epoch = self.current_epoch - self.pretrain_n_epochs
        return min(1.0, phase2_epoch / self.scheduled_sampling_T)

    def _get_n_steps(self, mv_v_part, mv_s_part, scalars, batch_idx, counts, use_ss=True):
        """Compute n_steps, optionally mixing GT and Head-N prediction (scheduled sampling).

        When use_ss=False (validation) always uses GT for comparable metrics across epochs.
        When alpha>0 a lightweight encode_only pass retrieves log_n_pred without running
        the decoder; the full forward is then called with the mixed n_steps value.
        """
        max_n_gt = int(counts.max().item())
        alpha = self._ss_alpha if use_ss else 0.0
        if alpha <= 0.0:
            return max(1, math.ceil(max_n_gt / self.d_step))
        with torch.no_grad():
            enc_out = self.model.forward_encode_only(mv_v_part, mv_s_part, scalars, batch_idx)
        n_pred = (torch.exp(enc_out["log_n_pred"].squeeze(-1)) - 1.0).clamp(min=1.0)
        max_n_pred = int(n_pred.max().item())
        max_n = max(1, int(round((1.0 - alpha) * max_n_gt + alpha * max_n_pred)))
        return max(1, math.ceil(max_n / self.d_step))

    @property
    def _is_global_zero(self):
        trainer = self.trainer
        return trainer is None or trainer.is_global_zero

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs(self, batch):
        data = build_batch(
            batch,
            use_scalar=self.use_scalar,
            use_one_hot=self.use_one_hot,
            use_time=self.use_time,
            use_log=False,
            use_energy=False,
        )
        mv_v_part = data["mv_v_part"].to(self.device)
        mv_s_part = data["mv_s_part"].to(self.device)
        scalars   = data["scalars"].to(self.device)
        batch_idx = data["batch_idx"].to(self.device)

        # Ground truth hit features for Hungarian loss (in normalised space)
        true_xyz = mv_v_part                           # (N, 3) — already on device
        true_k   = data["mv_s_part"].to(self.device)  # (N, 1) — k or zeros

        # Always use raw thr (not one-hot) for Hungarian loss
        true_thr = batch.thr.to(self.device)           # (N, 1)

        # Per-event hit counts for n_steps computation
        counts = torch.bincount(batch_idx, minlength=1)  # (B,)

        return mv_v_part, mv_s_part, scalars, batch_idx, true_xyz, true_k, true_thr, counts

    # ------------------------------------------------------------------
    # KL loss helper (same formula as train_autoencoder.py lines 69-72)
    # ------------------------------------------------------------------

    def _kl_loss(self, out):
        mu     = out.get("embed_mu")
        logvar = out.get("embed_logvar")
        if mu is None or logvar is None:
            return torch.tensor(0.0, device=self.device)
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def forward(self, mv_v_part, mv_s_part, scalars, batch_idx, n_steps=None):
        return self.model(mv_v_part, mv_s_part, scalars, batch_idx, n_steps=n_steps)

    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()

        if self.current_epoch == self.pretrain_n_epochs:
            self._unfreeze_decoder()
            wprint(
                f"[Epoch {self.current_epoch}] Phase 2 activated: decoder unfrozen. "
                f"Hungarian loss now active."
            )

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.perf_counter()

    def training_step(self, batch, batch_idx_arg):
        (mv_v_part, mv_s_part, scalars, batch_idx,
         true_xyz, true_k, true_thr, counts) = self._prepare_inputs(batch)

        batch_size = int(getattr(batch, "num_graphs", int(batch_idx.max().item()) + 1))
        log_n_target = torch.log(counts.float() + 1).unsqueeze(-1)  # (B, 1)

        log_kwargs = dict(on_step=True, on_epoch=False, sync_dist=True,
                          prog_bar=False, batch_size=batch_size)
        log_epoch_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True,
                                prog_bar=True, batch_size=batch_size)

        if not self._is_phase2:
            # ---- Phase 1: encoder + aggregation + VAE head + Head N ----
            out = self.model.forward_encode_only(mv_v_part, mv_s_part, scalars, batch_idx)
            loss_n   = F.mse_loss(out["log_n_pred"], log_n_target)
            loss_kl  = self._kl_loss(out)
            loss     = loss_n + self.beta_kl * loss_kl

            self.log("loss/train_batch",    loss,    **log_kwargs)
            self.log("loss/train_epoch",    loss,    **log_epoch_kwargs)
            self.log("loss/n_train",        loss_n,  **log_kwargs)
            self.log("loss/kl_train",       loss_kl, **log_kwargs)
        else:
            # ---- Phase 2: full forward + Hungarian loss ----
            n_steps = self._get_n_steps(
                mv_v_part, mv_s_part, scalars, batch_idx, counts, use_ss=True)

            out = self.forward(mv_v_part, mv_s_part, scalars, batch_idx, n_steps=n_steps)

            loss_n  = F.mse_loss(out["log_n_pred"], log_n_target)
            loss_kl = self._kl_loss(out)

            # w_k = 0 if use_scalar=False (depth not used by encoder)
            w_xyz = self.hungarian_w.get("w_xyz", 1.0)
            w_k   = self.hungarian_w.get("w_k",   1.0) if self.use_scalar else 0.0
            w_thr = self.hungarian_w.get("w_thr",  1.0)

            loss_h = hungarian_loss_batch(
                out["gen_xyz"], out["gen_k"], out["gen_thr"],
                true_xyz, true_k, true_thr, batch_idx,
                w_xyz=w_xyz, w_k=w_k, w_thr=w_thr,
            )

            loss = (
                self.lambda_hungarian * loss_h
                + self.lambda_n * loss_n
                + self.beta_kl * loss_kl
            )

            self.log("loss/train_batch",     loss,    **log_kwargs)
            self.log("loss/train_epoch",     loss,    **log_epoch_kwargs)
            self.log("loss/hungarian_train", loss_h,  **log_kwargs)
            self.log("loss/n_train",         loss_n,  **log_kwargs)
            self.log("loss/kl_train",        loss_kl, **log_kwargs)
            self.log("train/ss_alpha",       self._ss_alpha, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False, batch_size=batch_size)

        # Gradient and timing debug
        if self._is_global_zero and wandb is not None:
            step = int(self.global_step)
            if step % self.debug_cfg.get("debug_grad_step", 500) == 0:
                self._debug_grad_step = step
            if self._batch_start_time is not None:
                wandb.log({"train/batch_time_s": time.perf_counter() - self._batch_start_time})

        return loss

    def on_after_backward(self):
        if self._is_global_zero and wandb is not None and self._debug_grad_step is not None:
            _log_gradient_stats(self.model, self._debug_grad_step)
            self._debug_grad_step = None
        if self._ema is not None:
            self._ema.update()

    def on_train_epoch_end(self):
        if self._is_global_zero and wandb is not None and self._epoch_start_time is not None:
            wandb.log({"train/epoch_time_s": time.perf_counter() - self._epoch_start_time})

    def _validation_step_inner(self, batch):
        (mv_v_part, mv_s_part, scalars, batch_idx,
         true_xyz, true_k, true_thr, counts) = self._prepare_inputs(batch)

        batch_size   = int(getattr(batch, "num_graphs", int(batch_idx.max().item()) + 1))
        log_n_target = torch.log(counts.float() + 1).unsqueeze(-1)

        if not self._is_phase2:
            out    = self.model.forward_encode_only(mv_v_part, mv_s_part, scalars, batch_idx)
            loss_n = F.mse_loss(out["log_n_pred"], log_n_target)
            loss_kl = self._kl_loss(out)
            loss   = loss_n + self.beta_kl * loss_kl
        else:
            n_steps = self._get_n_steps(
                mv_v_part, mv_s_part, scalars, batch_idx, counts, use_ss=False)
            out     = self.forward(mv_v_part, mv_s_part, scalars, batch_idx, n_steps=n_steps)

            loss_n  = F.mse_loss(out["log_n_pred"], log_n_target)
            loss_kl = self._kl_loss(out)

            w_xyz = self.hungarian_w.get("w_xyz", 1.0)
            w_k   = self.hungarian_w.get("w_k",   1.0) if self.use_scalar else 0.0
            w_thr = self.hungarian_w.get("w_thr",  1.0)

            loss_h = hungarian_loss_batch(
                out["gen_xyz"], out["gen_k"], out["gen_thr"],
                true_xyz, true_k, true_thr, batch_idx,
                w_xyz=w_xyz, w_k=w_k, w_thr=w_thr,
            )
            loss = (
                self.lambda_hungarian * loss_h
                + self.lambda_n * loss_n
                + self.beta_kl * loss_kl
            )

        self.log("loss/val", loss, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True, batch_size=batch_size)
        self._val_losses_local.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        if self._ema is not None:
            with self._ema.average_parameters():
                return self._validation_step_inner(batch)
        return self._validation_step_inner(batch)

    def on_validation_epoch_end(self):
        self._val_losses_local.clear()

    # ------------------------------------------------------------------
    # Optimizer and scheduler — identical to LightningGATrRegressor
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.optimizer_cfg.get("weight_decay", 1e-4),
        )

        warmup_pct          = self.scheduler_cfg.get("warmup_pct", 0.0)
        warmup_start_factor = self.scheduler_cfg.get("warmup_start_factor", 0.1)
        interval            = self.scheduler_cfg.get("interval", "epoch")
        scheduler_type      = self.scheduler_cfg.get("type", "cosine")

        if interval == "step":
            total_steps  = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * warmup_pct)

            if scheduler_type == "cosine":
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=max(1, total_steps - warmup_steps),
                    eta_min=self.scheduler_cfg.get("lr_min", 1e-6),
                )
            else:
                main_sched = torch.optim.lr_scheduler.StepLR(
                    opt,
                    step_size=self.scheduler_cfg.get("decay_steps", 1000),
                    gamma=self.scheduler_cfg.get("decay_rate", 0.5),
                )

            if warmup_steps > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                sched = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sched, main_sched],
                    milestones=[warmup_steps],
                )
            else:
                sched = main_sched
        else:
            max_epochs     = self.max_epochs
            warmup_epochs  = int(max_epochs * warmup_pct)

            if scheduler_type == "cosine":
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=max(1, max_epochs - warmup_epochs),
                    eta_min=self.scheduler_cfg.get("lr_min", 1e-6),
                )
            else:
                main_sched = torch.optim.lr_scheduler.StepLR(
                    opt,
                    step_size=self.scheduler_cfg.get("decay_steps", 30),
                    gamma=self.scheduler_cfg.get("decay_rate", 0.5),
                )

            if warmup_epochs > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                sched = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sched, main_sched],
                    milestones=[warmup_epochs],
                )
            else:
                sched = main_sched

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": interval,
            },
        }

    # ------------------------------------------------------------------
    # EMA and checkpointing — same pattern as LightningGATrRegressor
    # ------------------------------------------------------------------

    def on_fit_start(self):
        if self.trainer.is_global_zero and wandb is not None and self.logger is not None:
            self.logger.watch(
                self.model,
                log="gradients",
                log_freq=self.debug_cfg.get("gradients_log_step", 500),
                log_graph=False,
            )
        if self.use_ema:
            try:
                from torch_ema import ExponentialMovingAverage
                self._ema = ExponentialMovingAverage(
                    self.model.parameters(), decay=self.ema_decay
                )
                if self._ema_state_to_load is not None:
                    self._ema.load_state_dict(self._ema_state_to_load)
                    self._ema_state_to_load = None
            except ImportError:
                import warnings
                warnings.warn("torch_ema not installed — EMA disabled.")
                self._ema = None

    def on_save_checkpoint(self, checkpoint):
        if self._ema is not None:
            checkpoint["ema_state_dict"] = self._ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema and "ema_state_dict" in checkpoint:
            self._ema_state_to_load = checkpoint["ema_state_dict"]

    def on_fit_end(self):
        if not self._is_global_zero:
            return
        final_path = os.path.join(self.output_path, "gatr_ar_autoencoder_final.pt")
        payload = {
            "epoch":            int(self.current_epoch),
            "global_step":      int(self.global_step),
            "model_state_dict": self.model.state_dict(),
        }
        if self.trainer.optimizers:
            payload["optimizer_state_dict"] = self.trainer.optimizers[0].state_dict()
        if self.trainer.lr_scheduler_configs:
            payload["scheduler_state_dict"] = (
                self.trainer.lr_scheduler_configs[0].scheduler.state_dict()
            )
        torch.save(payload, final_path)
        wprint(f"Modelo final guardado en {final_path}")
