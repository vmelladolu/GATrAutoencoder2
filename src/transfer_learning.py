import torch
import torch.nn as nn


class TransferClassifier(nn.Module):

    def __init__(self, autoencoder, latent_dim, n_classes=3):

        super().__init__()

        self.autoencoder = autoencoder

        # freeze encoder
        for p in self.autoencoder.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_classes)
        )

    def forward(
        self,
        mv_v_part,
        mv_s_part,
        scalars,
        batch
    ):

        with torch.no_grad():

            outputs = self.autoencoder(
                mv_v_part,
                mv_s_part,
                scalars,
                batch
            )

            z = outputs["aggregate_latent"]

        logits = self.classifier(z)

        return logits
