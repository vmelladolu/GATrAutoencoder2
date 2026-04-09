import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from .gatr_module import GATrBasicModule


class GATrAutoencoder(nn.Module):
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
        cfg_dec,
        cfg_agg={"type": "mean"}, # or "max" or None
        latent_s_channels=16,
        use_vae=False,
        event_embed_dim=32,
    ):
        super().__init__()
        self.use_vae = use_vae

        # Encoder: maps inputs -> latent scalars (and latent mv if needed)
        self.encoder = GATrBasicModule(
            hidden_mv_channels=cfg_enc["hidden_mv_channels"],
            hidden_s_channels=cfg_enc["hidden_s_channels"],
            num_blocks=cfg_enc["num_blocks"],
            in_s_channels=cfg_enc["in_s_channels"],
            in_mv_channels=cfg_enc["in_mv_channels"],
            out_mv_channels=cfg_enc["out_mv_channels"],
            dropout=cfg_enc["dropout"],
            out_s_channels=cfg_enc["out_s_channels"],
            mv_embedding_mode=cfg_enc.get("mv_embedding_mode", "single"),
        )
        self.cfg_agg = cfg_agg
        self.compressor = nn.Linear(cfg_enc["out_s_channels"] + 16, latent_s_channels) # Compress concatenated latent mv and scalar to a smaller latent space
        # Decoder: maps latent scalars (+ optional mv) -> reconstructed scalars
        self.decoder = GATrBasicModule(
            hidden_mv_channels=cfg_dec["hidden_mv_channels"],
            hidden_s_channels=cfg_dec["hidden_s_channels"],
            num_blocks=cfg_dec["num_blocks"],
            in_s_channels=cfg_dec["in_s_channels"],
            in_mv_channels=cfg_dec["in_mv_channels"],
            out_mv_channels=cfg_dec["out_mv_channels"],
            dropout=cfg_dec["dropout"],
            out_s_channels=cfg_dec["out_s_channels"],
        )
        out_mv_channels = cfg_enc["out_mv_channels"]
        out_s_channels = cfg_enc["out_s_channels"]
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

        projector_in = latent_s_channels + head_in
        # LayerNorm antes de los proyectores: evita que escalas distintas entre la parte local
        # (latent_compressed, 2D) y la parte global (aggregate_latent, 80D) creen gradientes
        # asimétricos para x, y, z.
        self.proj_norm = nn.LayerNorm(projector_in)
        self.coord_projector = nn.Linear(projector_in, 4)
        self.scalar_projector = nn.Linear(projector_in, cfg_dec["in_s_channels"])

        # Cabeza VAE opcional sobre aggregate_latent para estructurar el espacio de clustering.
        # No requiere etiquetas: regulariza el espacio hacia N(0,I) mediante pérdida KL.
        if use_vae:
            self.vae_norm   = nn.LayerNorm(head_in)
            self.vae_mu     = nn.Linear(head_in, event_embed_dim)
            self.vae_logvar = nn.Linear(head_in, event_embed_dim)
        else:
            self.vae_norm = self.vae_mu = self.vae_logvar = None
            
    @staticmethod
    def _reparameterize(mu, log_var):
        """Reparameterization trick: mu + eps * std, con eps ~ N(0,I)."""
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def encode(self, mv_v_part, mv_s_part, scalars, batch):
        mv_latent, s_latent, point_latent, scalar_latent = self.encoder(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            batch=batch,
        )
        return mv_latent, s_latent, point_latent, scalar_latent

    def decode(self, mv_latent, mv_s_part, s_latent, batch):
        mv_rec, s_rec, point_rec, scalar_rec = self.decoder(
            mv_v_part=mv_latent.squeeze(1),
            mv_s_part=mv_s_part,
            scalars=s_latent,
            batch=batch,
        )
        return mv_rec, s_rec, point_rec, scalar_rec

    def forward(self, mv_v_part, mv_s_part, scalars, batch):
        mv_latent, s_latent, point_latent, scalar_latent = self.encode(
            mv_v_part, mv_s_part, scalars, batch
        ) # encode the data
        
        # ----------------------------------------------
        #------------- LATENT MANIPULATION -------------
        # ----------------------------------------------
        # concatenate latent mv and scalar representations and compress to a smaller latent space
        latent_concat = torch.cat([mv_latent.squeeze(1), s_latent], dim=-1) # (N, 16 + F_s)
        latent_compressed = self.compressor(latent_concat) # (N, latent_s_channels)

        #----------------------------------------------
        #------------- AGGREGATION STEP ---------------
        #----------------------------------------------
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
                mv_latent_agg = scatter_mean(mv_flat, batch, dim=0)
                s_latent_agg = scatter_mean(s_latent, batch, dim=0)
            elif agg_type == "max":
                mv_latent_agg = scatter_max(mv_flat, batch, dim=0)[0]
                s_latent_agg = scatter_max(s_latent, batch, dim=0)[0]
            elif agg_type == "sum":
                mv_latent_agg = scatter_sum(mv_flat, batch, dim=0)
                s_latent_agg = scatter_sum(s_latent, batch, dim=0)
            else:
                raise ValueError(f"Aggregation type not supported: {agg_type}")

            aggregate_latent = torch.cat([mv_latent_agg, s_latent_agg], dim=-1)  # (B, out_mv_channels*16 + out_s_channels)
        #----------------------------------------------
        #------------- VAE HEAD (clustering) ----------
        #----------------------------------------------
        # Opcional: estructura el aggregate_latent para clustering sin etiquetas.
        # Con use_vae=False, event_embedding = aggregate_latent (sin cambios).
        if self.use_vae:
            h = self.vae_norm(aggregate_latent)
            embed_mu     = self.vae_mu(h)       # (B, event_embed_dim)
            embed_logvar = self.vae_logvar(h)   # (B, event_embed_dim)
            if self.training:
                event_embedding = self._reparameterize(embed_mu, embed_logvar)
            else:
                event_embedding = embed_mu      # determinístico en eval
        else:
            event_embedding = aggregate_latent  # (B, head_in) — usar directamente para clustering
            embed_mu = embed_logvar = None

        #----------------------------------------------
        #------------- DECODER INPUT PREPARATION ------
        #----------------------------------------------
        # create full latent rpr by concatenating compressed latent, aggregated scalar latent, and aggregated mv latent
        if agg_type == "attention":
            aggregate_latent_expanded = aggregate_latent[batch] # (N, num_seeds * D)
            latent_full_repr = torch.cat([latent_compressed, aggregate_latent_expanded], dim=-1) # (N, latent_s_channels + num_seeds * D)
        else:
            # expand mv_latent_agg and s_latent_agg to (N, 16) and (N, F_s) respectively for concatenation
            mv_latent_agg_expanded = mv_latent_agg[batch] # (N, 16)
            s_latent_agg_expanded = s_latent_agg[batch] # (N, F_s)
            latent_full_repr = torch.cat([latent_compressed, s_latent_agg_expanded, mv_latent_agg_expanded], dim=-1) # (B, latent_s_channels + F_s + 16)

        # LayerNorm antes de proyectar: iguala escalas entre latent_compressed (2D) y aggregate (80D)
        latent_full_repr_normed = self.proj_norm(latent_full_repr)

        # project to 3D coordinates for decoding
        mv_latent_agg_projected = self.coord_projector(latent_full_repr_normed)

        # project to s_in dimension for decoding
        s_latent_agg_projected = self.scalar_projector(latent_full_repr_normed)
        
        
        mv_rec, s_rec, point_rec, scalar_rec = self.decode(
            mv_latent_agg_projected[:, :3], mv_latent_agg_projected[:, 3:], s_latent_agg_projected, batch
        )

        return {
            "mv_latent": mv_latent,          # (N, 1, 16)
            "s_latent": s_latent,            # (N, 1, F_s)
            "point_latent": point_latent,    # (N, 3)
            "scalar_latent": scalar_latent,  # (N, 1)
            "aggregate_latent": aggregate_latent,  # (B, head_in) — candidato principal para clustering
            "event_embedding": event_embedding,    # (B, event_embed_dim) si VAE, else = aggregate_latent
            "embed_mu": embed_mu,            # (B, event_embed_dim) o None si use_vae=False
            "embed_logvar": embed_logvar,    # (B, event_embed_dim) o None si use_vae=False
            "mv_rec": mv_rec,
            "s_rec": s_rec,                  # the rest of the variables of the hit
            "point_rec": point_rec,          # coordinate of the detector
            "scalar_rec": scalar_rec,        # layer of the detector (for example)
        }
