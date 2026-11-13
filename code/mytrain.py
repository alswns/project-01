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
    n=0
    total_loss = 0
    total_recon = 0
    total_kl = 0
    for x in dataloader:
        x=x.to(device)
        x_hat,mu,logvar = model(x)
        loss, recon, kl = loss_function(x,x_hat,mu,logvar)
        optimizer.zero_grad()
        loss.backward()
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

