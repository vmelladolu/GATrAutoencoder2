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
        cfg_agg={"type": "mean"}, # or "max" or None
    ):
        super().__init__()

        # Encoder: maps inputs -> latent scalars (and latent mv if needed)
        self.encoder = GATrBasicModule(
            hidden_mv_channels=cfg_enc["hidden_mv_channels"],
            hidden_s_channels=cfg_enc["hidden_s_channels"],
            num_blocks=cfg_enc["num_blocks"],
            in_s_channels=cfg_enc["in_s_channels"],
            in_mv_channels=cfg_enc["in_mv_channels"],
            out_mv_channels=cfg_enc["out_mv_channels"],
            dropout=cfg_enc["dropout"],
            out_s_channels=cfg_enc["out_s_channels"]
        )
        self.cfg_agg = cfg_agg
        head_in = cfg_enc["out_s_channels"] + 16
        self.energy_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

        # Predict energy from concatenated latent mv and scalar
        # self.compressor = nn.Linear(cfg_enc["out_s_channels"] + 16, latent_s_channels) # Compress concatenated latent mv and scalar to a smaller latent space
        # self.coord_projector = nn.Linear(latent_s_channels + cfg_enc["out_s_channels"] + 16, 4) # Project latent mv to the same dimension as latent scalars for aggregation
        # self.scalar_projector = nn.Linear(latent_s_channels + cfg_enc["out_s_channels"] + 16, cfg_dec["in_s_channels"]) # Project latent scalars to the same dimension as input scalars for decoding
        # # Decoder: maps latent scalars (+ optional mv) -> reconstructed scalars
        # self.decoder = GATrBasicModule(
        #     hidden_mv_channels=cfg_dec["hidden_mv_channels"],
        #     hidden_s_channels=cfg_dec["hidden_s_channels"],
        #     num_blocks=cfg_dec["num_blocks"],
        #     in_s_channels=cfg_dec["in_s_channels"],
        #     in_mv_channels=cfg_dec["in_mv_channels"],
        #     out_mv_channels=cfg_dec["out_mv_channels"],
        #     dropout=cfg_dec["dropout"],
        #     out_s_channels=cfg_dec["out_s_channels"],
        # )

    def encode(self, mv_v_part, mv_s_part, scalars, batch):
        mv_latent, s_latent, point_latent, scalar_latent = self.encoder(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            batch=batch,
        )
        return mv_latent, s_latent, point_latent, scalar_latent

    def forward(self, mv_v_part, mv_s_part, scalars, batch):
        mv_latent, s_latent, _, _ = self.encode(
            mv_v_part, mv_s_part, scalars, batch
        ) # encode the data
        
        #----------------------------------------------
        #------------- AGGREGATION STEP ---------------
        #----------------------------------------------
        if self.cfg_agg.get("type", "mean") == "mean":
            mv_latent_agg = scatter_mean(mv_latent.squeeze(1), batch, dim=0) # (B, 16)
            s_latent_agg = scatter_mean(s_latent, batch, dim=0) # (B, F_s)
        elif self.cfg_agg.get("type") == "max":
            mv_latent_agg = scatter_max(mv_latent.squeeze(1), batch, dim=0) # (B, 16)
            s_latent_agg = scatter_max(s_latent, batch, dim=0) # (B, F_s)
        elif self.cfg_agg.get("type") == "sum":
            mv_latent_agg = scatter_sum(mv_latent.squeeze(1), batch, dim=0) # (B, 16)
            s_latent_agg = scatter_sum(s_latent, batch, dim=0) # (B, F_s)
          
        # aggregate event information by taking mean of latent representations for each event
        aggregate_latent = torch.cat([mv_latent_agg, s_latent_agg], dim=-1) # (B, 16 + F_s)
        # OPCIONAL
        # SE PODRÍA PONER OTRA CAPA LINEAL PARA PROYECTAR A OTRO ESPACIO
        
        energy_pred = self.energy_head(aggregate_latent) # (B, 1)

        return energy_pred.squeeze(1)
