"""Optimization settings: batch size, number of epochs, optimizer.

TODO: choose BATCH_SIZE and NUM_EPOCHS, and complete `get_optimizer`.
"""

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# Used by the notebook. The values below are only a starting point.
BATCH_SIZE = 64
NUM_EPOCHS = 400
LR = 0.005
EMA_DECAY = 0.995

def get_optimizer(model):
    """Create the optimizer that updates the parameters of `model`.

    The choice of the optimizer and of its settings (learning rate, ...) is
    yours.

    Args:
        model: the VAE defined in mymodel.py.

    Returns:
        a torch.optim.Optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # EMA copy of the model, updated by train_one_epoch after every step.
    optimizer.ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY)) if EMA_DECAY else None
    return optimizer
