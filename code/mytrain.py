"""Training of the VAE: loss function and one epoch of training.

TODO: complete `loss_function` and `train_one_epoch`.
"""

import math
import torch
import torch.nn.functional as F
from myoptim import NUM_EPOCHS, LR

K = 4

WARMUP = 5
LR_MIN = 0.02


def set_lr(optimizer, epoch):
    if epoch < WARMUP:
        f = (epoch + 1) / WARMUP
    else:
        f = LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP) / (NUM_EPOCHS - WARMUP)))
    for g in optimizer.param_groups:
        g["lr"] = LR * f


AUG_P = 0.5
AUG_FRAC = 0.9
AUG_ROT, AUG_SCALE, AUG_SHIFT, AUG_SHEAR = 8, 0.1, 1.0 / 16, 0.1


def augment(x):
    B, d = x.size(0), x.device
    a = (torch.rand(B, device=d) * 2 - 1) * AUG_ROT * math.pi / 180
    s = 1 + (torch.rand(B, device=d) * 2 - 1) * AUG_SCALE
    sh = (torch.rand(B, device=d) * 2 - 1) * AUG_SHEAR
    t = (torch.rand(B, 2, device=d) * 2 - 1) * AUG_SHIFT
    theta = torch.zeros(B, 2, 3, device=d)
    theta[:, 0, 0] = torch.cos(a) / s
    theta[:, 0, 1] = (-torch.sin(a) + sh) / s
    theta[:, 1, 0] = torch.sin(a) / s
    theta[:, 1, 1] = torch.cos(a) / s
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xa = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros").clamp(0, 1)
    keep = torch.rand(B, 1, 1, 1, device=d) >= AUG_P
    return torch.where(keep, x, xa)


def loss_function(x, x_hat, mu, logvar):
    """VAE loss (negative ELBO) of a batch, averaged over its images.

        loss  = recon + kl

        recon : reconstruction loss, the binary cross-entropy between x_hat
                and x, summed over the pixels of an image
        kl    : KL divergence KL( q(z|x) || p(z) ) with the prior
                p(z) = N(0, I), summed over the latent dimensions

    Both terms are averaged over the B images of the batch.

    Args:
        x:      images,                  shape (B, 1, 32, 32), values in [0, 1].
        x_hat:  decoded images,          shape (B, 1, 32, 32), values in [0, 1].
        mu:     mean of q(z|x),          shape (B, latent_dim).
        logvar: log-variance of q(z|x),  shape (B, latent_dim).

    Returns:
        (loss, recon, kl): three scalar tensors.
    """
    # TODO
    Batch_size = x.size(0)
    recon = F.binary_cross_entropy(x_hat,x,reduction="sum") / Batch_size
    #가우시안 KL 
    sigma = torch.exp(logvar/2)
    kl = torch.log(1.0 / sigma) + (sigma**2 + mu**2) / 2 - 0.5
    kl = kl.sum() / Batch_size
    loss = recon + kl
    return loss,recon,kl

def train_one_epoch(model, optimizer, dataloader, device):
    """Train `model` for one epoch, i.e. one pass over `dataloader`.

    For every mini-batch: move the images to `device`, run the model,
    compute the loss with `loss_function`, back-propagate, and update the
    parameters with `optimizer`. Put the model in training mode first.

    Args:
        model:      the VAE, already on `device`.
        optimizer:  the optimizer from myoptim.get_optimizer.
        dataloader: yields batches of images, shape (B, 1, 32, 32), values
                    in [0, 1] (there are no labels).
        device:     torch.device to compute on.

    Returns:
        dict {"loss": ..., "recon": ..., "kl": ...} of Python floats: the
        average of each term per image over the whole epoch.
    """
    model.train()
    epoch = optimizer.epoch
    use_aug = epoch < AUG_FRAC * NUM_EPOCHS
    set_lr(optimizer, epoch)
    n=0
    total_loss = 0
    total_recon = 0
    total_kl = 0
    for x in dataloader:
        x=x.to(device)
        if use_aug:
            x = augment(x)
        mu, logvar = model.encode(x)
        x, mu, logvar = x.repeat(K, 1, 1, 1), mu.repeat(K, 1), logvar.repeat(K, 1)
        z = model.reparameterize(mu, logvar)
        x_hat = model.decode(z)
        loss, recon, kl = loss_function(x,x_hat,mu,logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Batch_size = x.size(0)
        total_loss += loss.item() * Batch_size
        total_recon += recon.item() * Batch_size
        total_kl += kl.item() * Batch_size
        n += Batch_size

    optimizer.epoch += 1
    return {"loss": total_loss / n, "recon": total_recon / n, "kl": total_kl / n}

