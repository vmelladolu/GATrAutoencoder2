import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Pooling by Multihead Attention (PMA) for variable-length sequences
    grouped by batch index (PyG-style).

    Uses a learnable query (seed vector) that attends to all tokens of each
    event to produce a fixed-size summary per event.
    """

    def __init__(self, embed_dim, num_heads=4, num_seeds=1, dropout=0.0):
        super().__init__()
        self.num_seeds = num_seeds
        self.seed = nn.Parameter(torch.randn(1, num_seeds, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)

    def forward(self, x, batch):
        """
        Args:
            x:     (N_total, D)  tokens from all events
            batch: (N_total,)    event indices [0,0,..,1,1,..,2,...]
        Returns:
            out:   (B, num_seeds * D)  pooled vector per event
        """
        device = x.device
        counts = torch.bincount(batch)
        B = counts.shape[0]
        max_len = counts.max().item()
        D = x.shape[-1]

        # Build padded tensor and key padding mask (True = ignore position)
        padded = torch.zeros(B, max_len, D, device=device)
        key_padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=offsets[1:])

        for i in range(B):
            length = counts[i].item()
            padded[i, :length] = x[offsets[i]:offsets[i + 1]]
            key_padding_mask[i, :length] = False

        # Query: seed expanded to (B, num_seeds, D)
        query = self.seed.expand(B, -1, -1)
        query = self.norm_q(query)
        kv = self.norm_k(padded)

        # Cross-attention: query attends to event tokens
        out, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)

        # Flatten seeds: (B, num_seeds * D)
        return out.reshape(B, -1)
