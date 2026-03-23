import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from .gatr_module import GATrBasicModule


class GATrRegressor(nn.Module):
    """
    Autoencoder that uses GATr blocks for encoding and decoding.

    Expected inputs:
      - mv_v_part: (N, 3) point or vector
      - mv_s_part: (N, 1) geometric scalar
      - scalars: (N, F_in)
      - batch: (N,) batch indices
    """

    def __init__(
        self,
        cfg_enc,
        cfg_agg={"type": "sum"},
    ):
        super().__init__()
        self.cfg_agg = cfg_agg

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

        # ---- Attention pooling (Cambio 3) ----
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
            head_in = num_seeds * pool_dim
        else:
            self.attention_pool = None
            head_in = out_mv_channels * 16 + out_s_channels

        # Extra global features: N_hits
        if self.use_nhits_features:
            if self.nhits_mode == "per_threshold":
                head_in += 3  # log(N1+1), log(N2+1), log(N3+1)
            else:
                head_in += 1  # log(N_hits+1)

        # ---- Energy head (Cambio 6) ----
        head_cfg = cfg_agg.get("energy_head", {})
        head_hidden = head_cfg.get("hidden_dims", [256, 128, 64])
        head_activation = head_cfg.get("activation", "relu")
        head_use_layernorm = head_cfg.get("layernorm", True)
        head_dropout = head_cfg.get("dropout", 0.0)
        # Compatibilidad con checkpoints entrenados antes de que se añadiera
        # el guard `if head_dropout > 0`. Activar solo para cargar esos checkpoints.
        legacy_always_dropout = head_cfg.get("legacy_always_dropout", False)

        act_fn = nn.GELU if head_activation == "gelu" else nn.ReLU

        layers = []
        if head_use_layernorm:
            layers.append(nn.LayerNorm(head_in))

        prev_dim = head_in
        for dim in head_hidden:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(act_fn())
            if legacy_always_dropout or head_dropout > 0:
                layers.append(nn.Dropout(head_dropout))
            if head_use_layernorm and dim > 64:
                layers.append(nn.LayerNorm(dim))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.energy_head = nn.Sequential(*layers)

    def encode(self, mv_v_part, mv_s_part, scalars, batch):
        mv_latent, s_latent, point_latent, scalar_latent = self.encoder(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            batch=batch,
        )
        return mv_latent, s_latent, point_latent, scalar_latent

    def forward(self, mv_v_part, mv_s_part, scalars, batch, extra_global_features=None):
        mv_latent, s_latent, _, _ = self.encode(
            mv_v_part, mv_s_part, scalars, batch
        )
        # mv_latent: (N, out_mv_channels, 16)
        # s_latent:  (N, out_s_channels)

        agg_type = self.cfg_agg.get("type", "sum")

        if agg_type == "attention":
            # Flatten all MV channels then concatenate with scalars
            token_features = torch.cat(
                [mv_latent.view(mv_latent.size(0), -1), s_latent], dim=-1
            )  # (N, out_mv_channels*16 + out_s_channels)
            aggregate_latent = self.attention_pool(token_features, batch)  # (B, num_seeds * D)
        else:
            mv_flat = mv_latent.view(mv_latent.size(0), -1)  # (N, out_mv_channels*16)
            if agg_type == "mean":
                mv_agg = scatter_mean(mv_flat, batch, dim=0)
                s_agg = scatter_mean(s_latent, batch, dim=0)
            elif agg_type == "max":
                mv_agg = scatter_max(mv_flat, batch, dim=0)[0]
                s_agg = scatter_max(s_latent, batch, dim=0)[0]
            elif agg_type == "sum":
                mv_agg = scatter_sum(mv_flat, batch, dim=0)
                s_agg = scatter_sum(s_latent, batch, dim=0)
            else:
                raise ValueError(f"Aggregation type not supported: {agg_type}")

            aggregate_latent = torch.cat([mv_agg, s_agg], dim=-1)  # (B, out_mv_channels*16 + out_s_channels)

        # ---- Append N_hits features (Cambio 4) ----
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
                    # Fallback to total hits if per-threshold info not available
                    counts = torch.bincount(batch, minlength=1).float()
                    log_nhits = torch.log(counts + 1).unsqueeze(-1)
                    aggregate_latent = torch.cat([aggregate_latent, log_nhits], dim=-1)
            else:
                counts = torch.bincount(batch, minlength=1).float()
                log_nhits = torch.log(counts + 1).unsqueeze(-1)
                aggregate_latent = torch.cat([aggregate_latent, log_nhits], dim=-1)

        energy_pred = self.energy_head(aggregate_latent)  # (B, 1)
        return energy_pred.squeeze(1)
