import math
import os
import time

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from src.models.gatr_clf_regressor import GATrClassifierRegressor
from src.utils.batch_utils import build_batch
from src.utils.losses import reconstruction_loss
from src.utils.logging import wprint, _log_gradient_stats, _log_prediction_debug, _log_aggregation_debug
from src.utils.plots import _log_event_display, _log_regression_plots

try:
    import wandb
except ImportError:
    wandb = None


class LightningGATrClfRegressor(L.LightningModule):
    """
    Lightning module for GATrClassifierRegressor.

    Training is split into two phases:

    Phase 1 (epochs 0 .. phase1_epochs-1):
        - energy_head (and experts in MoE mode) FROZEN.
        - Trains encoder + aggregation + clf_head.
        - Loss: alpha_cls * CE(class_logits, bin_labels) + alpha_lb * KL_load_balance

    Phase 2 (epochs phase1_epochs .. max_epochs-1):
        - All parameters trainable.
        - Gumbel-Softmax annealing activates (if use_gumbel=True).
        - Loss: loss_reg + alpha_cls * CE + alpha_lb * KL_load_balance
    """

    def __init__(
        self,
        cfg_enc,
        cfg_agg,
        class_weights,
        use_scalar,
        use_one_hot,
        use_log,
        z_norm,
        stats,
        learning_rate,
        max_epochs,
        plot_every,
        output_path,
        n_bins,
        energy_min_gev,
        energy_max_gev,
        alpha_cls,
        alpha_lb=0.05,
        label_smoothing=0.0,
        phase1_epochs=20,
        use_gumbel=False,
        tau_start=1.0,
        tau_end=0.1,
        tau_anneal_epochs=50,
        use_time=False,
        scheduler_cfg=None,
        optimizer_cfg=None,
        debug_cfg=None,
        reg_ramp_epochs=10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "stats"])

        self.model = GATrClassifierRegressor(cfg_enc=cfg_enc, cfg_agg=cfg_agg, n_bins=n_bins)

        self.class_weights = class_weights
        self.use_scalar = use_scalar
        self.use_one_hot = use_one_hot
        self.use_time = use_time
        self.use_log = use_log
        self.z_norm = z_norm
        self.stats = stats
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.plot_every = plot_every
        self.output_path = output_path

        # Classification / bin parameters
        self.n_bins = n_bins
        self.alpha_cls = alpha_cls
        self.alpha_lb = alpha_lb
        self.label_smoothing = label_smoothing
        self.register_buffer("bin_edges", torch.linspace(energy_min_gev, energy_max_gev, n_bins + 1))

        # Two-phase training
        self.phase1_epochs = phase1_epochs
        self.reg_ramp_epochs = max(1, reg_ramp_epochs)

        # Gumbel-Softmax annealing
        self.use_gumbel = use_gumbel
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_anneal_epochs = tau_anneal_epochs

        # Optimizer / scheduler
        self.scheduler_cfg = scheduler_cfg if scheduler_cfg else {"type": "cosine"}
        self.optimizer_cfg = optimizer_cfg if optimizer_cfg else {}
        self.debug_cfg = debug_cfg if debug_cfg else {}
        self.loss_type = self.optimizer_cfg.get("loss_type", "mse")
        self.huber_delta = self.optimizer_cfg.get("huber_delta", 1.0)
        self.use_ema = self.optimizer_cfg.get("ema", False)
        self.ema_decay = self.optimizer_cfg.get("ema_decay", 0.999)
        self._ema = None
        self._ema_state_to_load = None

        self._batch_start_time = None
        self._epoch_start_time = None
        self._debug_grad_step = None

        # Validation accumulators
        self._val_true_local = []
        self._val_pred_local = []
        self._val_cls_true_local = []
        self._val_cls_pred_local = []

        # Freeze energy head at start (Phase 1)
        self._freeze_energy_head()

    # ------------------------------------------------------------------
    # Phase management (mirrors lightning_ar_trainer.py)
    # ------------------------------------------------------------------

    def _freeze_energy_head(self):
        targets = ["energy_head"] if not self.model.use_moe else ["experts"]
        for name in targets:
            module = getattr(self.model, name)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)

    def _unfreeze_energy_head(self):
        targets = ["energy_head"] if not self.model.use_moe else ["experts"]
        for name in targets:
            module = getattr(self.model, name)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(True)

    @property
    def _is_phase2(self):
        return self.current_epoch >= self.phase1_epochs

    @property
    def _is_global_zero(self):
        trainer = self.trainer
        return trainer is None or trainer.is_global_zero

    # ------------------------------------------------------------------
    # Gumbel temperature
    # ------------------------------------------------------------------

    def _get_tau(self):
        """
        Returns (tau, hard) for the current training step.
        - Phase 1: (None, False)  — plain softmax, energy head frozen anyway
        - Phase 2 with gumbel off: (None, False)  — plain softmax
        - Phase 2 with gumbel on:  cosine anneal from tau_start to tau_end over
          the first tau_anneal_epochs epochs of phase 2; hard=True throughout.
        """
        if not self._is_phase2 or not self.use_gumbel:
            return None, False

        phase2_epoch = self.current_epoch - self.phase1_epochs
        if self.tau_anneal_epochs <= 0 or phase2_epoch >= self.tau_anneal_epochs:
            return self.tau_end, True

        progress = phase2_epoch / self.tau_anneal_epochs  # 0 → 1
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
        tau = self.tau_end + (self.tau_start - self.tau_end) * cos_factor
        return tau, True

    # ------------------------------------------------------------------
    # Input preparation (identical to LightningGATrRegressor)
    # ------------------------------------------------------------------

    def _prepare_inputs(self, batch):
        data = build_batch(
            batch,
            use_scalar=self.use_scalar,
            use_one_hot=self.use_one_hot,
            use_time=self.use_time,
            use_log=False,   # log already applied at dataset init
            use_energy=True,
        )
        mv_v_part = data["mv_v_part"].to(self.device)
        mv_s_part = data["mv_s_part"].to(self.device)
        scalars = data["scalars"].to(self.device)
        batch_idx = data["batch_idx"].to(self.device)
        target = data["logenergy"].to(self.device)

        extra_global = {}
        for key in ("n_thr1", "n_thr2", "n_thr3"):
            val = data.get(key)
            if val is not None:
                extra_global[key] = val.to(self.device)

        return mv_v_part, mv_s_part, scalars, batch_idx, target, extra_global

    def _compute_bin_labels(self, target):
        """Compute integer bin labels from log-energy target."""
        energy_gev = torch.exp(target.view(-1))  # (B,)
        bin_labels = torch.bucketize(energy_gev, self.bin_edges)  # (B,) in [0, n_bins]
        return bin_labels.clamp(0, self.n_bins - 1)

    def _load_balance_loss(self, class_logits):
        """KL(mean_probs || uniform): penalises routing collapse."""
        mean_probs = torch.softmax(class_logits, dim=-1).mean(dim=0)  # (n_bins,)
        uniform = torch.full_like(mean_probs, 1.0 / self.n_bins)
        return F.kl_div(mean_probs.log(), uniform, reduction="sum")

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def forward(self, mv_v_part, mv_s_part, scalars, batch_idx,
                extra_global_features=None, tau=None, hard=False, detach_routing=False):
        return self.model(mv_v_part, mv_s_part, scalars, batch_idx,
                          extra_global_features=extra_global_features,
                          tau=tau, hard=hard, detach_routing=detach_routing)

    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()

        if self.current_epoch == self.phase1_epochs:
            self._unfreeze_energy_head()
            wprint(
                f"[Epoch {self.current_epoch}] Phase 2 activated: energy_head unfrozen. "
                f"Regression loss now active."
                + (" Gumbel-Softmax annealing starting." if self.use_gumbel else "")
            )

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.perf_counter()

    def training_step(self, batch, batch_idx):
        mv_v_part, mv_s_part, scalars, batch_idx_t, target, extra_global = self._prepare_inputs(batch)
        tau, hard = self._get_tau()

        # Fix 1: in Phase 2 detach routing so loss_reg does not corrupt the classifier.
        energy_pred, class_logits = self(
            mv_v_part, mv_s_part, scalars, batch_idx_t,
            extra_global_features=extra_global or None,
            tau=tau, hard=hard,
            detach_routing=self._is_phase2,
        )

        batch_size = int(getattr(batch, "num_graphs", target.shape[0]))
        bin_labels = self._compute_bin_labels(target)

        loss_cls = F.cross_entropy(class_logits, bin_labels,
                                   label_smoothing=self.label_smoothing)
        loss_lb = self._load_balance_loss(class_logits)

        if self._is_phase2:
            loss_reg = reconstruction_loss(
                energy_pred, target,
                class_weights=self.class_weights,
                loss_type=self.loss_type,
                huber_delta=self.huber_delta,
            )
            # Fix 2: ramp-up loss_reg linearly over reg_ramp_epochs to avoid a loss
            # spike at the phase transition while experts are still cold-starting.
            phase2_epoch = self.current_epoch - self.phase1_epochs
            reg_weight = min(1.0, (phase2_epoch + 1) / self.reg_ramp_epochs)
            loss = reg_weight * loss_reg + self.alpha_cls * loss_cls + self.alpha_lb * loss_lb
        else:
            loss_reg = torch.tensor(0.0, device=self.device)
            reg_weight = 0.0
            loss = self.alpha_cls * loss_cls + self.alpha_lb * loss_lb

        self.log("loss/train_reg",   loss_reg,   on_step=True,  on_epoch=False, sync_dist=True, prog_bar=False, batch_size=batch_size)
        self.log("loss/train_cls",   loss_cls,   on_step=True,  on_epoch=False, sync_dist=True, prog_bar=False, batch_size=batch_size)
        self.log("loss/train_lb",    loss_lb,    on_step=True,  on_epoch=False, sync_dist=True, prog_bar=False, batch_size=batch_size)
        self.log("loss/train_batch", loss,       on_step=True,  on_epoch=False, sync_dist=True, prog_bar=False, batch_size=batch_size)
        self.log("loss/train_epoch", loss,       on_step=False, on_epoch=True,  sync_dist=True, prog_bar=True,  batch_size=batch_size)
        self.log("train/reg_weight", reg_weight, on_step=True,  on_epoch=False, sync_dist=False, prog_bar=False, batch_size=batch_size)

        if tau is not None:
            self.log("train/gumbel_tau", tau, on_step=True, on_epoch=False)

        if self._is_global_zero and wandb is not None:
            step = int(self.global_step)
            if step % self.debug_cfg.get("debug_grad_step", 100) == 0:
                self._debug_grad_step = step
                if self._is_phase2:
                    _log_prediction_debug(energy_pred, target, step)
                _log_aggregation_debug(self.model, mv_v_part, mv_s_part, scalars, batch_idx_t, step)
            if self._is_phase2 and step % self.debug_cfg.get("debug_event_step", 500) == 0:
                _log_event_display(mv_v_part, batch_idx_t, energy_pred, target, self.use_log, step)
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
        mv_v_part, mv_s_part, scalars, batch_idx_v, target, extra_global = self._prepare_inputs(batch)
        tau, hard = self._get_tau()

        energy_pred, class_logits = self(
            mv_v_part, mv_s_part, scalars, batch_idx_v,
            extra_global_features=extra_global or None,
            tau=tau, hard=hard,
        )

        batch_size = int(getattr(batch, "num_graphs", target.shape[0]))
        bin_labels = self._compute_bin_labels(target)

        loss_cls = F.cross_entropy(class_logits, bin_labels,
                                   label_smoothing=self.label_smoothing)
        loss_lb = self._load_balance_loss(class_logits)

        if self._is_phase2:
            loss_reg = reconstruction_loss(
                energy_pred, target,
                class_weights=self.class_weights,
                loss_type=self.loss_type,
                huber_delta=self.huber_delta,
            )
            loss = loss_reg + self.alpha_cls * loss_cls + self.alpha_lb * loss_lb
        else:
            loss_reg = torch.tensor(0.0, device=self.device)
            loss = self.alpha_cls * loss_cls + self.alpha_lb * loss_lb

        self.log("loss/val",     loss,     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True,  batch_size=batch_size)
        self.log("loss/val_cls", loss_cls, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False, batch_size=batch_size)
        self.log("loss/val_reg", loss_reg, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False, batch_size=batch_size)

        self._val_true_local.append(target.cpu().detach())
        self._val_pred_local.append(energy_pred.cpu().detach())

        # Accumulate classification accuracy
        pred_bins = torch.argmax(class_logits, dim=-1)
        self._val_cls_true_local.append(bin_labels.cpu().detach())
        self._val_cls_pred_local.append(pred_bins.cpu().detach())

        return loss

    def validation_step(self, batch, batch_idx):
        if self._ema is not None:
            with self._ema.average_parameters():
                return self._validation_step_inner(batch)
        return self._validation_step_inner(batch)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._val_true_local.clear()
            self._val_pred_local.clear()
            self._val_cls_true_local.clear()
            self._val_cls_pred_local.clear()
            return

        # --- Gather regression predictions (DDP-safe, mirrors LightningGATrRegressor) ---
        all_true = []
        all_pred = []
        local_true = torch.cat(self._val_true_local, dim=0).view(-1).numpy() if self._val_true_local else np.array([])
        local_pred = torch.cat(self._val_pred_local, dim=0).view(-1).numpy() if self._val_pred_local else np.array([])

        # --- Gather classification labels ---
        all_cls_true = []
        all_cls_pred = []
        local_cls_true = torch.cat(self._val_cls_true_local, dim=0).numpy() if self._val_cls_true_local else np.array([])
        local_cls_pred = torch.cat(self._val_cls_pred_local, dim=0).numpy() if self._val_cls_pred_local else np.array([])

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_true = [None] * world_size
            gathered_pred = [None] * world_size
            gathered_cls_true = [None] * world_size
            gathered_cls_pred = [None] * world_size
            torch.distributed.all_gather_object(gathered_true, local_true)
            torch.distributed.all_gather_object(gathered_pred, local_pred)
            torch.distributed.all_gather_object(gathered_cls_true, local_cls_true)
            torch.distributed.all_gather_object(gathered_cls_pred, local_cls_pred)
            if self._is_global_zero:
                all_true  = [a for a in gathered_true  if a is not None and len(a) > 0]
                all_pred  = [a for a in gathered_pred  if a is not None and len(a) > 0]
                all_cls_true = [a for a in gathered_cls_true if a is not None and len(a) > 0]
                all_cls_pred = [a for a in gathered_cls_pred if a is not None and len(a) > 0]
        elif self._is_global_zero:
            all_true  = [local_true]  if len(local_true)     > 0 else []
            all_pred  = [local_pred]  if len(local_pred)     > 0 else []
            all_cls_true = [local_cls_true] if len(local_cls_true) > 0 else []
            all_cls_pred = [local_cls_pred] if len(local_cls_pred) > 0 else []

        if self._is_global_zero and wandb is not None:
            # Classification accuracy
            if all_cls_true and all_cls_pred:
                cls_true_np = np.concatenate(all_cls_true)
                cls_pred_np = np.concatenate(all_cls_pred)
                clf_acc = float((cls_pred_np == cls_true_np).mean())
                self.log("val/clf_accuracy", clf_acc)
                if wandb.run is not None:
                    wandb.log({"val/clf_accuracy": clf_acc, "epoch": int(self.current_epoch)})

            # Regression plots (only in phase 2)
            if self._is_phase2 and all_true and all_pred:
                epoch_id = int(self.current_epoch) + 1
                if (epoch_id % self.plot_every == 0) or (epoch_id == self.max_epochs):
                    all_true_np = np.concatenate(all_true)
                    all_pred_np = np.concatenate(all_pred)
                    if self.use_log:
                        e_true = np.exp(all_true_np)
                        e_reco = np.exp(all_pred_np)
                    else:
                        e_true = all_true_np
                        e_reco = all_pred_np
                    _log_regression_plots(e_true, e_reco, step=int(self.global_step))

        self._val_true_local.clear()
        self._val_pred_local.clear()
        self._val_cls_true_local.clear()
        self._val_cls_pred_local.clear()

    # ------------------------------------------------------------------
    # Optimizer (identical to LightningGATrRegressor)
    # ------------------------------------------------------------------

    def _build_biphase_lambda(self, total_units, phase1_units):
        """
        Returns a lr_lambda function for LambdaLR implementing two independent
        warmup + cosine cycles, one per training phase.

        All returned values are multipliers on the base LR (self.learning_rate).

        Phase 1  [0, phase1_units):
            warmup : 0 .. w1            → start1  →  1.0
            cosine : w1 .. phase1_units → 1.0     →  eta_min1 / base_lr

        Phase 2  [phase1_units, total_units):
            warmup : phase1_units .. phase1_units+w2  → start2*max_factor → max_factor
            cosine : phase1_units+w2 .. total_units   → max_factor        → eta_min2 / base_lr

        Config keys (scheduler.phase1 / scheduler.phase2):
            warmup_pct          – fraction of the phase used for linear warmup
            warmup_start_factor – LR multiplier at the start of warmup
            lr_min              – absolute LR floor at the end of the cosine
            lr_max_factor       – (phase2 only) peak LR = base_lr * lr_max_factor
        """
        base_lr = self.learning_rate
        p1_cfg = self.scheduler_cfg.get("phase1", {})
        p2_cfg = self.scheduler_cfg.get("phase2", {})

        phase2_units = max(1, total_units - phase1_units)

        # Phase 1 parameters
        w1 = int(phase1_units * p1_cfg.get("warmup_pct", 0.1))
        start1 = p1_cfg.get("warmup_start_factor", 0.1)
        eta_min1 = p1_cfg.get("lr_min", 1e-4) / base_lr

        # Phase 2 parameters
        max_factor = p2_cfg.get("lr_max_factor", 0.3)
        w2 = int(phase2_units * p2_cfg.get("warmup_pct", 0.05))
        start2 = p2_cfg.get("warmup_start_factor", 0.1) * max_factor
        eta_min2 = p2_cfg.get("lr_min", 1e-5) / base_lr

        # Pre-compute boundary indices
        cosine1_start = w1
        cosine1_len = max(1, phase1_units - w1)
        warmup2_start = phase1_units
        cosine2_start = phase1_units + w2
        cosine2_len = max(1, total_units - cosine2_start)

        def lr_lambda(t):
            if t < cosine1_start:                          # Phase 1 warmup
                frac = t / max(1, w1)
                return start1 + (1.0 - start1) * frac
            elif t < phase1_units:                         # Phase 1 cosine
                progress = (t - cosine1_start) / cosine1_len
                cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return eta_min1 + (1.0 - eta_min1) * cos
            elif t < cosine2_start:                        # Phase 2 warmup
                frac = (t - warmup2_start) / max(1, w2)
                return start2 + (max_factor - start2) * frac
            else:                                          # Phase 2 cosine
                progress = (t - cosine2_start) / cosine2_len
                cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return eta_min2 + (max_factor - eta_min2) * cos

        return lr_lambda

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.optimizer_cfg.get("weight_decay", 1e-4),
        )

        interval = self.scheduler_cfg.get("interval", "epoch")
        use_biphase = ("phase1" in self.scheduler_cfg and "phase2" in self.scheduler_cfg)

        if use_biphase:
            # Fix 3: independent warmup+cosine schedule for each training phase.
            if interval == "step":
                total_units = self.trainer.estimated_stepping_batches
                steps_per_epoch = max(1, total_units // self.max_epochs)
                phase1_units = self.phase1_epochs * steps_per_epoch
            else:
                total_units = self.max_epochs
                phase1_units = self.phase1_epochs

            lr_lambda = self._build_biphase_lambda(total_units, phase1_units)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        else:
            # Legacy single-phase schedule (backward compatible)
            warmup_pct = self.scheduler_cfg.get("warmup_pct", 0.0)
            warmup_start_factor = self.scheduler_cfg.get("warmup_start_factor", 0.1)
            scheduler_type = self.scheduler_cfg.get("type", "cosine")

            if interval == "step":
                total_steps = self.trainer.estimated_stepping_batches
                warmup_steps = int(total_steps * warmup_pct)

                if scheduler_type == "cosine":
                    cosine_steps = max(1, total_steps - warmup_steps)
                    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=cosine_steps,
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
                        opt, start_factor=warmup_start_factor, end_factor=1.0,
                        total_iters=warmup_steps,
                    )
                    sched = torch.optim.lr_scheduler.SequentialLR(
                        opt, schedulers=[warmup_sched, main_sched], milestones=[warmup_steps],
                    )
                else:
                    sched = main_sched
            else:
                max_epochs = self.max_epochs
                warmup_epochs = int(max_epochs * warmup_pct)

                if scheduler_type == "cosine":
                    cosine_t_max = max(1, max_epochs - warmup_epochs)
                    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=cosine_t_max,
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
                        opt, start_factor=warmup_start_factor, end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    sched = torch.optim.lr_scheduler.SequentialLR(
                        opt, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs],
                    )
                else:
                    sched = main_sched

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": interval},
        }

    # ------------------------------------------------------------------
    # Checkpointing / EMA (identical to LightningGATrRegressor)
    # ------------------------------------------------------------------

    def on_fit_start(self):
        if self.trainer.is_global_zero and wandb is not None:
            self.logger.watch(
                self.model,
                log="gradients",
                log_freq=self.debug_cfg.get("gradients_log_step", 500),
                log_graph=False,
            )
        if self.use_ema:
            try:
                from torch_ema import ExponentialMovingAverage
                self._ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)
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
        final_model_path = os.path.join(self.output_path, "gatr_clf_regressor_final.pt")
        payload = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
        }
        if self.trainer.optimizers:
            payload["optimizer_state_dict"] = self.trainer.optimizers[0].state_dict()
        if self.trainer.lr_scheduler_configs:
            payload["scheduler_state_dict"] = self.trainer.lr_scheduler_configs[0].scheduler.state_dict()
        torch.save(payload, final_model_path)
        wprint(f"Modelo final guardado en {final_model_path}")
