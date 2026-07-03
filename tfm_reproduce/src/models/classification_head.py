import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Cabeza MLP de clasificación sobre un vector latente por evento.

    Pensada para transfer-learning: se entrena SOLA encima de un autoencoder
    GATr congelado, mapeando ``aggregate_latent`` o ``event_embedding`` a logits
    de clase (electron / pion / muon).

    Args:
        in_dim:       dimensión del latente de entrada (p.ej. ~80 para
                      aggregate_latent, 32 para event_embedding del VAE).
        n_classes:    número de clases de salida.
        hidden_dim:   anchura de la capa oculta.
        dropout:      probabilidad de dropout antes de la capa final.
        num_layers:   número de bloques ocultos (≥1).
    """

    def __init__(self, in_dim, n_classes=3, hidden_dim=128, dropout=0.2,
                 num_layers=1):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes

        layers = [nn.LayerNorm(in_dim)]
        d = in_dim
        for _ in range(max(1, num_layers)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, latent):
        """latent: (B, in_dim) -> logits: (B, n_classes)."""
        return self.net(latent)
