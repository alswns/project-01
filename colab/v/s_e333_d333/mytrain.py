"""Training of the VAE: loss function and one epoch of training.

TODO: complete `loss_function` and `train_one_epoch`.
"""

import torch
import torch.nn.functional as F


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

import math
from myoptim import NUM_EPOCHS
AUG_P = 0.5
AUG_FRAC = 0.8      # augment only during the first AUG_FRAC of the epochs
NOISE = 0.0        # std of the Gaussian noise added to the encoder input (target stays clean)
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
    ema = getattr(optimizer, "ema", None)
    if ema is not None and hasattr(optimizer, "live_state"):
        model.load_state_dict(optimizer.live_state)   # back to the trained weights

    model.train()
    optimizer.epoch = getattr(optimizer, 'epoch', -1) + 1
    use_aug = AUG_P > 0 and optimizer.epoch < AUG_FRAC * NUM_EPOCHS
    from myoptim import LR as _LR
    _e, _sch, _W, _min, _KW = optimizer.epoch, 'cos', 5, 0.02, 0
    if _sch == "cos":
        _f = (_e + 1) / _W if _e < _W else _min + (1 - _min) * 0.5 * (1 + math.cos(math.pi * (_e - _W) / max(1, NUM_EPOCHS - _W)))
        for g in optimizer.param_groups: g["lr"] = _LR * _f
    beta = min(1.0, (_e + 1) / _KW) if _KW else 1.0
    n=0
    total_loss = 0
    total_recon = 0
    total_kl = 0
    for x in dataloader:
        x=x.to(device)
        if use_aug:
            x = augment(x)
        xin = (x + NOISE * torch.randn_like(x)).clamp(0, 1) if NOISE else x
        mu, logvar = model.encode(xin)
        x_hat = model.decode(model.reparameterize(mu, logvar))
        loss, recon, kl = loss_function(x,x_hat,mu,logvar)
        optimizer.zero_grad()
        (recon + beta * kl).backward()
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)

        Batch_size = x.size(0)
        total_loss += loss.item() * Batch_size
        total_recon += recon.item() * Batch_size
        total_kl += kl.item() * Batch_size
        n += Batch_size

    # Between epochs (and after training) the model holds the EMA weights,
    # so the saved model, the samples and the evaluation all use them.
    if ema is not None:
        optimizer.live_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.module.state_dict())

    return {"loss": total_loss / n, "recon": total_recon / n, "kl": total_kl / n}

