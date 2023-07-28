"""Autoencoder model implementation."""

from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, enc_in_dim, enc_out_dim):
        super(Autoencoder, self).__init__()
        self.enc_in_dim = enc_in_dim
        self.enc_out_dim = enc_out_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.enc_in_dim, self.enc_out_dim), nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.enc_out_dim, self.enc_in_dim), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x
