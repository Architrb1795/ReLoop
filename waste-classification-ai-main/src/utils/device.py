"""Device selection — used by every training/inference script so the same
code runs unmodified on a CPU-only laptop or a CUDA-enabled Colab GPU.
"""
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
