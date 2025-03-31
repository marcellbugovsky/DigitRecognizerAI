# src/utils/device.py
import torch

def get_device():
    """Gets the default device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()