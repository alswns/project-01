"""Variational autoencoder (VAE).

TODO: complete the class `VAE`.

You are free to design the architecture (type and number of layers, their
sizes, the latent dimension), but keep the interface described in the
docstrings: the notebook and the other files rely on it.

Notation
    B      : number of images in a batch
    x      : images,                   shape (B, 1, 32, 32), values in [0, 1]
    z      : latent variables,         shape (B, latent_dim)
    mu     : mean of q(z|x),           shape (B, latent_dim)
    logvar : log-variance of q(z|x),   shape (B, latent_dim)
    x_hat  : decoded images,           shape (B, 1, 32, 32), values in [0, 1]
"""

import torch
import torch.nn as nn

LATENT_DIM = 16
RES_BLOCKS = 2   # ResBlocks per resolution (1 or 2)


class ResBlock(nn.Module):
    """x + f(x), f = (GroupNorm -> ReLU -> Conv3x3) x 2; keeps the shape."""

    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(
            nn.GroupNorm(8, c), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1),
            nn.GroupNorm(8, c), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.f(x)


def res(c):
    return [ResBlock(c) for _ in range(RES_BLOCKS)]


class VAE(nn.Module):
    def __init__(self):
        """Define the layers of the encoder and of the decoder.

        The notebook creates the model with `VAE()`, so every argument you
        add needs a default value.

        Must set `self.latent_dim` (int, at least 2): the dimension of z.
        """
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   nn.GroupNorm(8, 32),  nn.ReLU(), *res(32),    # 16x16
            nn.Conv2d(32, 64, 4, 2, 1),  nn.GroupNorm(8, 64),  nn.ReLU(), *res(64),    # 8x8
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(), *res(128),   # 4x4
            nn.Flatten(),
        )
        self.head_mu = nn.Linear(128 * 4 * 4, self.latent_dim)
        self.head_logvar = nn.Linear(128 * 4 * 4, self.latent_dim)

        self.decoder_align = nn.Linear(self.latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            *res(128),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.ReLU(), *res(64),  # 8x8
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1),  nn.GroupNorm(8, 32), nn.ReLU(), *res(32),  # 16x16
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 1, 3, 1, 1),   nn.Sigmoid(),                           # 32x32
        )

    def encode(self, x):
        """Encoder: parameters of q(z|x) = N(z; mu, diag(exp(logvar))).

        Args:
            x: images, shape (B, 1, 32, 32).

        Returns:
            (mu, logvar), each of shape (B, latent_dim).
        """
        latent = self.encoder(x)
        mu = self.head_mu(latent)
        logvar = self.head_logvar(latent)
        return (mu,logvar)
        
        

    def reparameterize(self, mu, logvar):
        """Draw z ~ N(mu, diag(exp(logvar))) with the reparameterization
        trick, so that gradients can flow back to mu and logvar.

        Args:
            mu, logvar: shape (B, latent_dim).

        Returns:
            z, shape (B, latent_dim).
        """
        
        # log(sigma 제곱) -> 2log(sigma) /2 후 exp -> std
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std*eps

    def decode(self, z):
        """Decoder: pixel-wise mean of p(x|z).

        Args:
            z: latent variables, shape (B, latent_dim).

        Returns:
            x_hat, shape (B, 1, 32, 32), values in [0, 1].
        """
        latent = self.decoder_align(z).view(-1, 128, 4, 4)
        
        return self.decoder(latent)
        

    def forward(self, x):
        """Encode x, draw z from q(z|x), and decode z.

        Args:
            x: images, shape (B, 1, 32, 32).

        Returns:
            (x_hat, mu, logvar).
        """

        mu,logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        x_hat = self.decode(z)
        return (x_hat, mu, logvar)
        
