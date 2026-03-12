import torch
import torch.nn as nn
from gatr.interface import embed_point, embed_scalar, extract_scalar, extract_point
from gatr import GATr, SelfAttentionConfig, MLPConfig
from xformers.ops.fmha import BlockDiagonalMask
from torch_scatter import scatter_mean


class GATrBasicModule(nn.Module):
    """
    GATr wrapper:
      - calorimeter hits → GA points   (x,y,z)
      - tracker hits     → GA vectors (normalized momentum)
      - geometric scalar = det_type_norm (goes to embed_scalar)
      - extra scalars (E, p_mod) passed directly to scalar MLP
    """

    def __init__(
        self,
        hidden_mv_channels=32,
        hidden_s_channels=64,
        num_blocks=2,
        in_s_channels=2,   # E, p_mod
        in_mv_channels=1,
        out_mv_channels=1,
        attention: SelfAttentionConfig = SelfAttentionConfig(),
        mlp: MLPConfig = MLPConfig(),
        dropout=0.1,
        out_s_channels=None,
        post_dropout=0.0,
        mv_embedding_mode="single",
        checkpoint=None,
    ):
        super().__init__()
        if out_s_channels is None:
            out_s_channels = hidden_s_channels

        # Validate in_mv_channels matches the embedding mode
        expected_mv_channels = 2 if mv_embedding_mode == "centroid" else 1
        assert in_mv_channels == expected_mv_channels, (
            f"in_mv_channels={in_mv_channels} does not match mv_embedding_mode='{mv_embedding_mode}' "
            f"(expected {expected_mv_channels})"
        )
        self.mv_embedding_mode = mv_embedding_mode

        gatr_kwargs = dict(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=attention,
            mlp=mlp,
            dropout_prob=dropout,
        )
        if checkpoint is not None:
            gatr_kwargs["checkpoint"] = checkpoint
        self.gatr = GATr(**gatr_kwargs)

        # Post-encoder dropout applied only on scalar output (preserves equivariance)
        self.scalar_dropout = nn.Dropout(post_dropout)

    # ---------------------------------------------
    # Embedding geométrico (punto o vector)
    # ---------------------------------------------
    def build_geom_embedding(self, mv_v_part, mv_s_part, batch):
        # mv_v_part: (N, 3) for points or (N, 3) for vectors
        # mv_s_part: (N, 1) for scalars (e.g., layer type)
        mv_vec = embed_point(mv_v_part)       # (N, 16)
        mv_scalar = embed_scalar(mv_s_part)   # (N, 16)

        if self.mv_embedding_mode == "single":
            embedded_geom = (mv_vec + mv_scalar).unsqueeze(1)  # (N, 1, 16)
        elif self.mv_embedding_mode == "centroid":
            # Channel 0: absolute position + scalar
            ch0 = (mv_vec + mv_scalar).unsqueeze(1)  # (N, 1, 16)
            # Channel 1: relative position (pos - centroid per event)
            centroid = scatter_mean(mv_v_part, batch, dim=0)  # (B, 3)
            pos_rel = mv_v_part - centroid[batch]              # (N, 3)
            ch1 = embed_point(pos_rel).unsqueeze(1)            # (N, 1, 16)
            embedded_geom = torch.cat([ch0, ch1], dim=1)       # (N, 2, 16)
        else:
            raise ValueError(f"Unknown mv_embedding_mode: {self.mv_embedding_mode}")

        return embedded_geom

    # ---------------------------------------------
    def build_attention_mask(self, batch):
        """ Create a block diagonal attention mask based on the batch indices.
        It only allows attention between elements of the same batch (i.e., same event).

        Args:
            batch (torch.Tensor): Batch indices for each element.

        Returns:
            BlockDiagonalMask: A block diagonal attention mask.
        """
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch.long()).tolist()
        )

    # ---------------------------------------------
    def forward(self, mv_v_part, mv_s_part, scalars, batch, embedded_geom=None):
        if embedded_geom is None:
            embedded_geom = self.build_geom_embedding(mv_v_part, mv_s_part, batch)
        else:
            assert embedded_geom.shape[-1] == 16, "Embedded geom last dim must be 16"
        mask = self.build_attention_mask(batch)

        mv_out, scalar_out = self.gatr(
            embedded_geom,           # (N, C_mv, 16)
            scalars=scalars,         # (N, F_in)
            attention_mask=mask
        )
        # mv_out: (N, out_mv_channels, 16)

        # Apply dropout only on scalars — never on mv (preserves equivariance)
        mv_out_final = mv_out                                    # (N, out_mv_channels, 16)
        scalar_out_final = self.scalar_dropout(scalar_out)       # (N, F_out)

        # Extract geometric quantities from first mv channel for convenience
        mv0 = mv_out_final[:, 0, :]                              # (N, 16)
        point = extract_point(mv0)                               # (N, 3)
        scalar = extract_scalar(mv0.unsqueeze(1))                # (N, 1, 1)
        scalar = scalar.view(-1, 1)                              # (N, 1)

        return mv_out_final, scalar_out_final, point, scalar