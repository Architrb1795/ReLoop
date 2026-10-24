"""Reproducibility utilities."""
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and (if installed) PyTorch.

    Called once at the start of any script/notebook that involves randomness
    (splitting, augmentation, weight initialization) so results are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
