import lightning as L
from src.models.gatr_regressor import GATrRegressor
from src.utils.batch_utils import build_batch
import time
from src.utils.losses import reconstruction_loss
from src.utils.logging import wprint, _log_gradient_stats, _log_prediction_debug, _log_aggregation_debug
from src.utils.plots import _log_event_display, _log_regression_plots
import wandb
import torch
import numpy as np
import os

class LightningGATrRegressor(L.LightningModule):
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
        use_time=False,
        scheduler_cfg=None,
        optimizer_cfg=None,
        debug_cfg=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "stats"])
        self.model = GATrRegressor(cfg_enc=cfg_enc, cfg_agg=cfg_agg)
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
        self._val_true_local = []
        self._val_pred_local = []
        self._debug_grad_step = None

    def forward(self, mv_v_part, mv_s_part, scalars, batch_idx, extra_global_features=None):
        return self.model(mv_v_part, mv_s_part, scalars, batch_idx,
                          extra_global_features=extra_global_features)

    @property
    def _is_global_zero(self):
        trainer = self.trainer
        return trainer is None or trainer.is_global_zero

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

    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.perf_counter()

    def training_step(self, batch, batch_idx):
        mv_v_part, mv_s_part, scalars, batch_idx_t, target, extra_global = self._prepare_inputs(batch)
        outputs = self(mv_v_part, mv_s_part, scalars, batch_idx_t,
                       extra_global_features=extra_global or None)
        loss = reconstruction_loss(outputs, target, class_weights=self.class_weights,
                                    loss_type=self.loss_type, huber_delta=self.huber_delta)
        batch_size = int(getattr(batch, "num_graphs", target.shape[0]))

        self.log("loss/train_batch", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False, batch_size=batch_size)
        self.log("loss/train_epoch", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch_size)

        if self._is_global_zero and wandb is not None:
            step = int(self.global_step)
            if step % self.debug_cfg.get("debug_grad_step", 100) == 0:
                self._debug_grad_step = step
                _log_prediction_debug(outputs, target, step)
                _log_aggregation_debug(self.model, mv_v_part, mv_s_part, scalars, batch_idx_t, step)
            if step % self.debug_cfg.get("debug_event_step", 500) == 0:
                _log_event_display(mv_v_part, batch_idx_t, outputs, target, self.use_log, step)

            if self._batch_start_time is not None:
                batch_time_s = time.perf_counter() - self._batch_start_time
                wandb.log({"train/batch_time_s": batch_time_s})

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
        outputs = self(mv_v_part, mv_s_part, scalars, batch_idx_v,
                       extra_global_features=extra_global or None)
        loss = reconstruction_loss(outputs, target, class_weights=self.class_weights,
                                    loss_type=self.loss_type, huber_delta=self.huber_delta)
        batch_size = int(getattr(batch, "num_graphs", target.shape[0]))
        self.log("loss/val", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch_size)
        self._val_true_local.append(target.cpu().detach())
        self._val_pred_local.append(outputs.cpu().detach())
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
            return

        all_true = []
        all_pred = []
        local_true = torch.cat(self._val_true_local, dim=0).view(-1).numpy() if self._val_true_local else np.array([])
        local_pred = torch.cat(self._val_pred_local, dim=0).view(-1).numpy() if self._val_pred_local else np.array([])

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_true = [None for _ in range(torch.distributed.get_world_size())]
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_true, local_true)
            torch.distributed.all_gather_object(gathered_pred, local_pred)
            if self._is_global_zero:
                for arr in gathered_true:
                    if arr is not None and len(arr) > 0:
                        all_true.append(arr)
                for arr in gathered_pred:
                    if arr is not None and len(arr) > 0:
                        all_pred.append(arr)
        elif self._is_global_zero:
            all_true = [local_true] if len(local_true) > 0 else []
            all_pred = [local_pred] if len(local_pred) > 0 else []

        if self._is_global_zero and all_true and all_pred and wandb is not None:
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

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     scheduler = self._build_scheduler(optimizer)
    #     interval = self.scheduler_cfg.get("interval", "epoch")
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": interval,
    #             "name": "lr_scheduler",
    #         },
    #     }
        
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.optimizer_cfg.get("weight_decay", 1e-4),
        )

        warmup_pct = self.scheduler_cfg.get("warmup_pct", 0.0)
        warmup_start_factor = self.scheduler_cfg.get("warmup_start_factor", 0.1)
        interval = self.scheduler_cfg.get("interval", "epoch")
        scheduler_type = self.scheduler_cfg.get("type", "cosine")

        if interval == "step":
            # Step-level scheduling: use estimated_stepping_batches from Lightning
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * warmup_pct)

            if scheduler_type == "cosine":
                cosine_steps = max(1, total_steps - warmup_steps)
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=cosine_steps,
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
            # Epoch-level scheduling (original behaviour)
            max_epochs = self.max_epochs
            warmup_epochs = int(max_epochs * warmup_pct)

            if scheduler_type == "cosine":
                cosine_t_max = max(1, max_epochs - warmup_epochs)
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=cosine_t_max,
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
            }
        }

    def _build_scheduler(self, optimizer):
        """Construye el scheduler según la configuración.
        
        Tipos soportados:
            - "cosine": CosineAnnealingLR
            - "step": StepLR
        """
        scheduler_type = self.scheduler_cfg.get("type", "cosine").lower()
        
        if scheduler_type == "cosine":
            eta_min = self.scheduler_cfg.get("eta_min", 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=eta_min,
            )
            wprint(f"Scheduler: CosineAnnealingLR (T_max={self.max_epochs}, eta_min={eta_min})")
        elif scheduler_type == "step":
            step_size = self.scheduler_cfg.get("step_size", 10)
            gamma = self.scheduler_cfg.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )
            wprint(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
        else:
            raise ValueError(f"Tipo de scheduler no soportado: {scheduler_type}. Usa 'cosine' o 'step'.")
        
        return scheduler

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
                warnings.warn("torch_ema not installed — EMA disabled. Install with: pip install torch_ema")
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
        final_model_path = os.path.join(self.output_path, "gatr_regressor_final.pt")
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
