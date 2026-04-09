import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from .gatr_module import GATrBasicModule


def _build_mlp(in_dim, hidden_dims, activation="gelu", layernorm=True, dropout=0.0,
               legacy_always_dropout=False):
    """Build a small MLP: [LayerNorm] → (Linear → Act → [Dropout] → [LayerNorm])* → Linear(1)."""
    act_fn = nn.GELU if activation == "gelu" else nn.ReLU
    layers = []
    if layernorm:
        layers.append(nn.LayerNorm(in_dim))
    prev_dim = in_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(act_fn())
        if legacy_always_dropout or dropout > 0:
            layers.append(nn.Dropout(dropout))
        if layernorm and dim > 64:
            layers.append(nn.LayerNorm(dim))
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


class GATrClassifierRegressor(nn.Module):
    """
    GATr-based regressor that first classifies the event into an energy bin and
    then uses that classification signal to aid energy regression.

    Architecture:
        GATrBasicModule [encoder]
        → scatter / AttentionPooling [aggregation]
        → [optional N_hits features]
        → clf_head MLP  → class_logits  (B, n_bins)
        → class_probs = softmax(class_logits)   [or Gumbel-Softmax if tau provided]
        → CONCAT MODE:  concat([agg_features, class_probs]) → energy_head MLP → (B,)
          MOE MODE:     sum_k(class_probs_k * expert_k(agg_features))            → (B,)

    Forward returns: (energy_pred: (B,), class_logits: (B, n_bins))
    """

    def __init__(
        self,
        cfg_enc,
        cfg_agg={"type": "sum"},
        n_bins: int = 10,
    ):
        super().__init__()
        self.cfg_agg = cfg_agg
        self.n_bins = n_bins

        # ---- Encoder (identical to GATrRegressor) ----
        mv_embedding_mode = cfg_enc.get("mv_embedding_mode", "single")
        checkpoint_cfg = cfg_enc.get("checkpoint", None)

        self.encoder = GATrBasicModule(
            hidden_mv_channels=cfg_enc["hidden_mv_channels"],
            hidden_s_channels=cfg_enc["hidden_s_channels"],
            num_blocks=cfg_enc["num_blocks"],
            in_s_channels=cfg_enc["in_s_channels"],
            in_mv_channels=cfg_enc["in_mv_channels"],
            out_mv_channels=cfg_enc["out_mv_channels"],
            dropout=cfg_enc["dropout"],
            out_s_channels=cfg_enc["out_s_channels"],
            post_dropout=cfg_enc.get("post_dropout", 0.0),
            mv_embedding_mode=mv_embedding_mode,
            checkpoint=checkpoint_cfg,
        )

        out_mv_channels = cfg_enc["out_mv_channels"]
        out_s_channels = cfg_enc["out_s_channels"]

        # ---- Aggregation (identical to GATrRegressor) ----
        self.use_nhits_features = cfg_agg.get("use_nhits_features", False)
        self.nhits_mode = cfg_agg.get("nhits_mode", "total")

        if cfg_agg.get("type") == "attention":
            from .attention_pooling import AttentionPooling
            pool_dim = out_mv_channels * 16 + out_s_channels
            num_seeds = cfg_agg.get("num_seeds", 1)
            self.attention_pool = AttentionPooling(
                embed_dim=pool_dim,
                num_heads=cfg_agg.get("num_heads", 4),
                num_seeds=num_seeds,
                dropout=cfg_agg.get("dropout", 0.0),
            )
            D_agg = num_seeds * pool_dim
        else:
            self.attention_pool = None
            D_agg = out_mv_channels * 16 + out_s_channels

        if self.use_nhits_features:
            if self.nhits_mode == "per_threshold":
                D_agg += 3
            else:
                D_agg += 1

        self.D_agg = D_agg

        # ---- Classification head: D_agg → n_bins ----
        clf_cfg = cfg_agg.get("clf_head", {})
        clf_hidden_dim = clf_cfg.get("hidden_dim", 128)
        clf_activation = clf_cfg.get("activation", "gelu")
        clf_layernorm = clf_cfg.get("layernorm", True)
        clf_dropout = clf_cfg.get("dropout", 0.0)
        clf_act_fn = nn.GELU if clf_activation == "gelu" else nn.ReLU

        clf_layers = []
        if clf_layernorm:
            clf_layers.append(nn.LayerNorm(D_agg))
        clf_layers.append(nn.Linear(D_agg, clf_hidden_dim))
        clf_layers.append(clf_act_fn())
        if clf_dropout > 0:
            clf_layers.append(nn.Dropout(clf_dropout))
        clf_layers.append(nn.Linear(clf_hidden_dim, n_bins))
        self.clf_head = nn.Sequential(*clf_layers)

        # ---- Regression head: concat or MoE ----
        clf_ext = cfg_agg.get("clf", {})  # fallback: top-level clf block may not be here
        self.use_moe = clf_ext.get("use_moe", False)

        head_cfg = cfg_agg.get("energy_head", {})
        head_hidden = head_cfg.get("hidden_dims", [256, 128, 64])
        head_activation = head_cfg.get("activation", "relu")
        head_use_layernorm = head_cfg.get("layernorm", True)
        head_dropout = head_cfg.get("dropout", 0.0)
        legacy_always_dropout = head_cfg.get("legacy_always_dropout", False)

        if self.use_moe:
            expert_hidden = clf_ext.get("expert_hidden_dims", [64, 32])
            self.experts = nn.ModuleList([
                _build_mlp(D_agg, expert_hidden, head_activation,
                           head_use_layernorm, head_dropout, legacy_always_dropout)
                for _ in range(n_bins)
            ])
            self.energy_head = None
            # Clamp expert outputs to a physically meaningful range (log-energy space).
            # Prevents unbounded expert magnitudes from exploding routing gradients.
            expert_clamp = clf_ext.get("expert_output_clamp", None)
            self._expert_clamp = tuple(expert_clamp) if expert_clamp is not None else None
        else:
            # Input: D_agg + n_bins (concatenation of agg_features and class_probs)
            reg_in = D_agg + n_bins
            self.energy_head = _build_mlp(reg_in, head_hidden, head_activation,
                                          head_use_layernorm, head_dropout, legacy_always_dropout)
            self.experts = None

    def encode(self, mv_v_part, mv_s_part, scalars, batch):
        """Identical to GATrRegressor.encode."""
        mv_latent, s_latent, point_latent, scalar_latent = self.encoder(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            batch=batch,
        )
        return mv_latent, s_latent, point_latent, scalar_latent

    def aggregate(self, mv_latent, s_latent, batch, extra_global_features=None):
        """
        Aggregates per-hit latent features to per-event representations.
        Mirrors the aggregation block of GATrRegressor.forward.

        Returns:
            agg_features: (B, D_agg)
        """
        agg_type = self.cfg_agg.get("type", "sum")

        if agg_type == "attention":
            token_features = torch.cat(
                [mv_latent.view(mv_latent.size(0), -1), s_latent], dim=-1
            )  # (N, D_token)
            aggregate_latent = self.attention_pool(token_features, batch)  # (B, num_seeds*D_token)
        else:
            mv_flat = mv_latent.view(mv_latent.size(0), -1)
            if agg_type == "mean":
                mv_agg = scatter_mean(mv_flat, batch, dim=0)
                s_agg = scatter_mean(s_latent, batch, dim=0)
            elif agg_type == "max":
                mv_agg = scatter_max(mv_flat, batch, dim=0)[0]
                s_agg = scatter_max(s_latent, batch, dim=0)[0]
            else:  # sum
                mv_agg = scatter_sum(mv_flat, batch, dim=0)
                s_agg = scatter_sum(s_latent, batch, dim=0)
            aggregate_latent = torch.cat([mv_agg, s_agg], dim=-1)

        # Append N_hits features (identical to GATrRegressor)
        if self.use_nhits_features:
            if self.nhits_mode == "per_threshold" and extra_global_features is not None:
                n1 = extra_global_features.get("n_thr1")
                n2 = extra_global_features.get("n_thr2")
                n3 = extra_global_features.get("n_thr3")
                if n1 is not None and n2 is not None and n3 is not None:
                    log_n1 = torch.log(n1.float() + 1).unsqueeze(-1)
                    log_n2 = torch.log(n2.float() + 1).unsqueeze(-1)
                    log_n3 = torch.log(n3.float() + 1).unsqueeze(-1)
                    aggregate_latent = torch.cat([aggregate_latent, log_n1, log_n2, log_n3], dim=-1)
                else:
                    counts = torch.bincount(batch, minlength=1).float()
                    log_nhits = torch.log(counts + 1).unsqueeze(-1)
                    aggregate_latent = torch.cat([aggregate_latent, log_nhits], dim=-1)
            else:
                counts = torch.bincount(batch, minlength=1).float()
                log_nhits = torch.log(counts + 1).unsqueeze(-1)
                aggregate_latent = torch.cat([aggregate_latent, log_nhits], dim=-1)

        return aggregate_latent  # (B, D_agg)

    def forward(
        self,
        mv_v_part,
        mv_s_part,
        scalars,
        batch,
        extra_global_features=None,
        tau=None,
        hard=False,
        detach_routing=False,
    ):
        """
        Args:
            tau:             If not None, use Gumbel-Softmax with this temperature.
            hard:            If True (with tau), use straight-through hard Gumbel-Softmax.
            detach_routing:  If True (MoE mode, Phase 2), stop gradients from loss_reg
                             flowing back through class_probs into the classifier. The
                             classifier is then trained exclusively via loss_cls, while
                             loss_reg trains only the experts and encoder.

        Returns:
            energy_pred:   (B,)
            class_logits:  (B, n_bins)
        """
        mv_latent, s_latent, _, _ = self.encode(mv_v_part, mv_s_part, scalars, batch)
        agg_features = self.aggregate(mv_latent, s_latent, batch, extra_global_features)
        # agg_features: (B, D_agg)

        class_logits = self.clf_head(agg_features)  # (B, n_bins)

        if tau is not None:
            class_probs = F.gumbel_softmax(class_logits, tau=tau, hard=hard)
        else:
            class_probs = torch.softmax(class_logits, dim=-1)
        # class_probs: (B, n_bins)

        if self.use_moe:
            # Each expert: (B, D_agg) → (B, 1); stack → (B, n_bins)
            expert_preds = torch.stack(
                [expert(agg_features).squeeze(1) for expert in self.experts], dim=1
            )  # (B, n_bins)
            if self._expert_clamp is not None:
                expert_preds = expert_preds.clamp(self._expert_clamp[0], self._expert_clamp[1])
            # Fix 1: detach routing in Phase 2 so loss_reg does not corrupt the classifier.
            routing = class_probs.detach() if detach_routing else class_probs
            energy_pred = (routing * expert_preds).sum(dim=1)  # (B,)
        else:
            combined = torch.cat([agg_features, class_probs], dim=-1)  # (B, D_agg + n_bins)
            energy_pred = self.energy_head(combined).squeeze(1)  # (B,)

        return energy_pred, class_logits
