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

LATENT_DIM = 24
ENC_BLOCKS = [1, 2, 3]   # BasicBlocks per layer: 32->16, 16->8, 8->4
DEC_BLOCKS = [3, 2, 1]   # BasicBlocks per layer: 4->8, 8->16, 16->32


class BasicBlock(nn.Module):
    """ResNet block: ReLU( GN(conv(ReLU(GN(conv(x))))) + shortcut(x) ).
    The shortcut is a 1x1 conv + GN when the shape changes."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1), nn.GroupNorm(8, c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1),      nn.GroupNorm(8, c_out),
        )
        if stride == 1 and c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride), nn.GroupNorm(8, c_out))
        nn.init.zeros_(self.f[-1].weight)   # zero-init: the residual branch starts at 0

    def forward(self, x):
        return torch.relu(self.f(x) + self.shortcut(x))


def make_layer(c_in, c_out, n, resample):
    """n blocks; the first one changes the channels and the resolution.
    "down": the first block uses stride 2. "up": nearest x2 upsampling first."""
    layer = nn.Sequential()
    if resample == "up":
        layer.append(nn.Upsample(scale_factor=2))
        layer.append(BasicBlock(c_in, c_out))
    else:
        layer.append(BasicBlock(c_in, c_out, stride=2))
    for _ in range(n - 1):
        layer.append(BasicBlock(c_out, c_out))
    return layer


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
            nn.Conv2d(1, 32, kernel_size=3, padding= 1), nn.GroupNorm(8, 32), nn.ReLU(),   # stem, 32x32
            make_layer(32, 32, ENC_BLOCKS[0], "down"),                   # 16x16
            make_layer(32, 64, ENC_BLOCKS[1], "down"),                   # 8x8
            make_layer(64, 128, ENC_BLOCKS[2], "down"),                  # 4x4
            nn.Flatten(),
        )
        self.head_mu = nn.Linear(128 * 4 * 4, self.latent_dim)
        self.head_logvar = nn.Linear(128 * 4 * 4, self.latent_dim)

        self.decoder_align = nn.Linear(self.latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            make_layer(128, 64, DEC_BLOCKS[0], "up"),                    # 8x8
            make_layer(64, 32, DEC_BLOCKS[1], "up"),                     # 16x16
            make_layer(32, 32, DEC_BLOCKS[2], "up"),                     # 32x32
            nn.Conv2d(32, 1, kernel_size=3, padding= 1), nn.Sigmoid(),
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
        
