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

LATENT_DIM = 32


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1), nn.GroupNorm(8, c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1), nn.GroupNorm(8, c_out),
        )
        if stride == 1 and c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride), nn.GroupNorm(8, c_out))
        nn.init.zeros_(self.f[-1].weight)

    def forward(self, x):
        return torch.relu(self.f(x) + self.shortcut(x))


def down(c_in, c_out):
    return nn.Sequential(ResBlock(c_in, c_out, 2), ResBlock(c_out, c_out))


def up(c_in, c_out):
    return nn.Sequential(nn.Upsample(scale_factor=2), ResBlock(c_in, c_out), ResBlock(c_out, c_out))


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
            nn.Conv2d(1, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.ReLU(),
            down(32, 32),
            down(32, 64),
            down(64, 128),
            nn.Flatten(),
        )
        self.head_mu = nn.Linear(128 * 4 * 4, self.latent_dim)
        self.head_logvar = nn.Linear(128 * 4 * 4, self.latent_dim)

        self.decoder_align = nn.Linear(self.latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            up(128, 64),
            up(64, 32),
            up(32, 32),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid(),
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
        
