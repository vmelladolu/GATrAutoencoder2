import math

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_sum

from .gatr_module import GATrBasicModule


class GATrARAutoencoder(nn.Module):
    """
    Autoregressive autoencoder based on GATr.

    Architecture:
        Encoder  →  Aggregation  →  VAE head (optional)  →  Head N
                                                         →  Autoregressive Decoder

    The decoder generates hits d at a time, using accumulated context:
        step k input: [seed_token] + [all k*d previously generated hits]
        output: last d tokens → new d hits (xyz, k, thr)

    Expected inputs:
      - mv_v_part: (N, 3) hit coordinates
      - mv_s_part: (N, 1) geometric scalar (depth k, or zeros)
      - scalars:   (N, F_in) threshold/features
      - batch:     (N,) batch indices
    """

    def __init__(
        self,
        cfg_enc,
        cfg_dec,
        cfg_agg=None,
        cfg_n_head=None,
        d=4,
        use_vae=False,
        event_embed_dim=32,
    ):
        super().__init__()
        if cfg_agg is None:
            cfg_agg = {"type": "mean"}
        if cfg_n_head is None:
            cfg_n_head = {}

        self.d = d
        self.use_vae = use_vae
        self.cfg_agg = cfg_agg

        # ------------------------------------------------------------------
        # Encoder (identical to GATrRegressor)
        # ------------------------------------------------------------------
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
            mv_embedding_mode=cfg_enc.get("mv_embedding_mode", "single"),
            checkpoint=cfg_enc.get("checkpoint", None),
        )

        # ------------------------------------------------------------------
        # Aggregation (same 4 methods as GATrRegressor / GATrAutoencoder)
        # ------------------------------------------------------------------
        out_mv_channels = cfg_enc["out_mv_channels"]
        out_s_channels = cfg_enc["out_s_channels"]
        agg_type = cfg_agg.get("type", "mean")

        if agg_type == "attention":
            from .attention_pooling import AttentionPooling
            pool_dim = out_mv_channels * 16 + out_s_channels
            num_seeds = cfg_agg.get("num_seeds", 1)
            self.attention_pool = AttentionPooling(
                embed_dim=pool_dim,
                num_heads=cfg_agg.get("num_heads", 4),
                num_seeds=num_seeds,
                dropout=cfg_agg.get("dropout", 0.0),
            )
            agg_dim = num_seeds * pool_dim
        else:
            self.attention_pool = None
            agg_dim = out_mv_channels * 16 + out_s_channels

        # ------------------------------------------------------------------
        # VAE head — same pattern as GATrAutoencoder lines 82-93
        # Regularises aggregate_latent toward N(0,I) via KL loss,
        # enabling unsupervised clustering on event_embedding.
        # ------------------------------------------------------------------
        if use_vae:
            self.vae_norm   = nn.LayerNorm(agg_dim)
            self.vae_mu     = nn.Linear(agg_dim, event_embed_dim)
            self.vae_logvar = nn.Linear(agg_dim, event_embed_dim)
            cond_dim = event_embed_dim
        else:
            self.vae_norm = self.vae_mu = self.vae_logvar = None
            cond_dim = agg_dim

        # ------------------------------------------------------------------
        # Head N — same builder pattern as energy_head in GATrRegressor
        # Predicts log(N_hits + 1); output is used to set n_steps at inference.
        # ------------------------------------------------------------------
        head_hidden     = cfg_n_head.get("hidden_dims", [128, 64])
        head_activation = cfg_n_head.get("activation", "relu")
        head_layernorm  = cfg_n_head.get("layernorm", True)
        head_dropout    = cfg_n_head.get("dropout", 0.0)
        act_fn = nn.GELU if head_activation == "gelu" else nn.ReLU

        layers = []
        if head_layernorm:
            layers.append(nn.LayerNorm(cond_dim))
        prev_dim = cond_dim
        for dim in head_hidden:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(act_fn())
            if head_dropout > 0:
                layers.append(nn.Dropout(head_dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.head_n = nn.Sequential(*layers)

        # ------------------------------------------------------------------
        # Seed token projectors — same pattern as GATrAutoencoder coord/scalar
        # projectors (lines 77-78). event_embedding → initial decoder token.
        # ------------------------------------------------------------------
        dec_in_s = cfg_dec["in_s_channels"]
        self.seed_norm     = nn.LayerNorm(cond_dim)
        self.seed_proj_xyz = nn.Linear(cond_dim, 3)
        self.seed_proj_mvs = nn.Linear(cond_dim, 1)   # geometric scalar (k)
        self.seed_proj_s   = nn.Linear(cond_dim, dec_in_s)

        # ------------------------------------------------------------------
        # Autoregressive Decoder
        # mv_embedding_mode="single" (in_mv_channels=1) — no centroid for
        # generated hits (no well-defined centroid across decode steps).
        # ------------------------------------------------------------------
        self.decoder = GATrBasicModule(
            hidden_mv_channels=cfg_dec["hidden_mv_channels"],
            hidden_s_channels=cfg_dec["hidden_s_channels"],
            num_blocks=cfg_dec["num_blocks"],
            in_s_channels=dec_in_s,
            in_mv_channels=cfg_dec.get("in_mv_channels", 1),
            out_mv_channels=cfg_dec["out_mv_channels"],
            dropout=cfg_dec["dropout"],
            out_s_channels=cfg_dec["out_s_channels"],
            mv_embedding_mode=cfg_dec.get("mv_embedding_mode", "single"),
        )

        # ------------------------------------------------------------------
        # Output heads for generated hits
        # ------------------------------------------------------------------
        dec_out_s = cfg_dec["out_s_channels"]
        self.thr_head = nn.Sequential(
            nn.LayerNorm(dec_out_s),
            nn.Linear(dec_out_s, 1),
        )
        # Projects decoder output back to decoder input space for the next step
        self.output_to_input_proj = nn.Sequential(
            nn.LayerNorm(dec_out_s),
            nn.Linear(dec_out_s, dec_in_s),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(self, mv_latent, s_latent, batch):
        """Aggregate per-hit representations to per-event. Returns (B, agg_dim)."""
        agg_type = self.cfg_agg.get("type", "mean")
        if agg_type == "attention":
            token_features = torch.cat(
                [mv_latent.view(mv_latent.size(0), -1), s_latent], dim=-1
            )
            return self.attention_pool(token_features, batch)

        mv_flat = mv_latent.view(mv_latent.size(0), -1)
        if agg_type == "mean":
            mv_agg = scatter_mean(mv_flat, batch, dim=0)
            s_agg  = scatter_mean(s_latent, batch, dim=0)
        elif agg_type == "max":
            mv_agg = scatter_max(mv_flat, batch, dim=0)[0]
            s_agg  = scatter_max(s_latent, batch, dim=0)[0]
        elif agg_type == "sum":
            mv_agg = scatter_sum(mv_flat, batch, dim=0)
            s_agg  = scatter_sum(s_latent, batch, dim=0)
        else:
            raise ValueError(f"Aggregation type not supported: {agg_type}")

        return torch.cat([mv_agg, s_agg], dim=-1)

    def _vae_head(self, aggregate_latent):
        """Apply optional VAE head. Returns (event_embedding, mu, logvar)."""
        if not self.use_vae:
            return aggregate_latent, None, None

        h      = self.vae_norm(aggregate_latent)
        mu     = self.vae_mu(h)
        logvar = self.vae_logvar(h)
        if self.training:
            embedding = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            embedding = mu  # deterministic at eval for clean clustering
        return embedding, mu, logvar

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_encode_only(self, mv_v_part, mv_s_part, scalars, batch):
        """
        Run encoder + aggregation + VAE head + Head N only (no decoder).
        Used in phase 1 training to avoid unnecessary computation.
        """
        mv_latent, s_latent, _, _ = self.encoder(mv_v_part, mv_s_part, scalars, batch)
        aggregate_latent          = self._aggregate(mv_latent, s_latent, batch)
        event_embedding, mu, logvar = self._vae_head(aggregate_latent)
        log_n_pred = self.head_n(event_embedding)

        return {
            "aggregate_latent": aggregate_latent,
            "event_embedding":  event_embedding,
            "embed_mu":         mu,
            "embed_logvar":     logvar,
            "log_n_pred":       log_n_pred,
        }

    def decode_all(self, event_embedding, n_steps):
        """
        Autoregressive decoding loop.

        At step k the context is [seed_token] + [all k*d previously generated hits].
        All events in the batch share the same context length at each step, so
        reshaping to (B, seq_len, *) is valid.

        Args:
            event_embedding: (B, cond_dim)
            n_steps: number of autoregressive steps; generates n_steps * d hits total

        Returns:
            gen_xyz: (B, n_gen, 3)
            gen_k:   (B, n_gen, 1)
            gen_thr: (B, n_gen, 1)
        """
        B      = event_embedding.shape[0]
        d      = self.d
        device = event_embedding.device

        # Project aggregate → seed token
        h        = self.seed_norm(event_embedding)
        seed_xyz = self.seed_proj_xyz(h).unsqueeze(1)   # (B, 1, 3)
        seed_mvs = self.seed_proj_mvs(h).unsqueeze(1)   # (B, 1, 1)
        seed_s   = self.seed_proj_s(h).unsqueeze(1)     # (B, 1, dec_in_s)

        # Accumulation buffers grow by d tokens each step
        acc_xyz = seed_xyz   # (B, seq_len, 3)
        acc_mvs = seed_mvs   # (B, seq_len, 1)
        acc_s   = seed_s     # (B, seq_len, dec_in_s)

        gen_xyz_steps = []
        gen_k_steps   = []
        gen_thr_steps = []

        for _ in range(n_steps):
            seq_len = acc_xyz.shape[1]   # 1 + step * d

            # Flatten buffers for GATrBasicModule
            mv_v_flat = acc_xyz.reshape(B * seq_len, 3)
            mv_s_flat = acc_mvs.reshape(B * seq_len, 1)
            s_flat    = acc_s.reshape(B * seq_len, -1)
            batch_dec = torch.arange(B, device=device).repeat_interleave(seq_len)

            # Decoder forward
            _, s_out, point_out, scalar_out = self.decoder(
                mv_v_flat, mv_s_flat, s_flat, batch_dec
            )

            # Extract last d outputs per event (reshape trick is valid because
            # all events have identical seq_len at every step)
            new_xyz = point_out.view(B, seq_len, 3)[:, -d:, :]       # (B, d, 3)
            new_k   = scalar_out.view(B, seq_len, 1)[:, -d:, :]      # (B, d, 1)
            s_r     = s_out.view(B, seq_len, -1)[:, -d:, :]          # (B, d, dec_out_s)

            new_thr  = self.thr_head(s_r)                             # (B, d, 1)
            new_s_in = self.output_to_input_proj(s_r)                 # (B, d, dec_in_s)

            gen_xyz_steps.append(new_xyz)
            gen_k_steps.append(new_k)
            gen_thr_steps.append(new_thr)

            # Append to context buffers for next step
            acc_xyz = torch.cat([acc_xyz, new_xyz],  dim=1)
            acc_mvs = torch.cat([acc_mvs, new_k],    dim=1)
            acc_s   = torch.cat([acc_s,   new_s_in], dim=1)

        gen_xyz = torch.cat(gen_xyz_steps, dim=1)   # (B, n_gen, 3)
        gen_k   = torch.cat(gen_k_steps,   dim=1)   # (B, n_gen, 1)
        gen_thr = torch.cat(gen_thr_steps, dim=1)   # (B, n_gen, 1)

        return gen_xyz, gen_k, gen_thr

    def forward(self, mv_v_part, mv_s_part, scalars, batch, n_steps=None):
        """
        Full forward pass: encode → VAE → head N → autoregressive decode.

        Args:
            n_steps: number of autoregressive steps. If None, derived from head N
                     prediction (suitable for inference). For training, caller
                     should provide n_steps = ceil(max_N_in_batch / d).

        Returns dict with keys:
            log_n_pred, aggregate_latent, event_embedding,
            embed_mu, embed_logvar, gen_xyz, gen_k, gen_thr
        """
        mv_latent, s_latent, _, _ = self.encoder(mv_v_part, mv_s_part, scalars, batch)
        aggregate_latent          = self._aggregate(mv_latent, s_latent, batch)
        event_embedding, mu, logvar = self._vae_head(aggregate_latent)
        log_n_pred = self.head_n(event_embedding)

        if n_steps is None:
            n_pred = torch.exp(log_n_pred.squeeze(-1).detach()) - 1
            n_steps = max(1, math.ceil(float(n_pred.max().item()) / self.d))

        gen_xyz, gen_k, gen_thr = self.decode_all(event_embedding, n_steps)

        return {
            "log_n_pred":       log_n_pred,         # (B, 1)
            "aggregate_latent": aggregate_latent,    # (B, agg_dim)
            "event_embedding":  event_embedding,     # (B, cond_dim)
            "embed_mu":         mu,                  # (B, event_embed_dim) or None
            "embed_logvar":     logvar,              # (B, event_embed_dim) or None
            "gen_xyz":          gen_xyz,             # (B, n_gen, 3)
            "gen_k":            gen_k,               # (B, n_gen, 1)
            "gen_thr":          gen_thr,             # (B, n_gen, 1)
        }
