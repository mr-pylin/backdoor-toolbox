import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


def calculate_mean_and_std(dataset: Dataset, batch_size: int = 256) -> Tuple[list[float], list[float]]:
    """
    Computes the per-channel mean and std of a dataset in a memory-efficient way.

    Args:
        dataset (Dataset): The input dataset (expects samples of shape [C, H, W]).
        batch_size (int): Batch size for processing.

    Returns:
        Tuple[list[float], list[float]]: Per-channel mean and std.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_pixels = 0
    sum_channels = None
    sum_squared = None

    for batch in loader:
        imgs = batch[0]  # (B, C, H, W)
        B, C, H, W = imgs.shape
        if sum_channels is None:
            sum_channels = torch.zeros(C)
            sum_squared = torch.zeros(C)

        imgs = imgs.view(B, C, -1)  # (B, C, H*W)
        n_pixels += B * H * W
        sum_channels += imgs.sum(dim=(0, 2))
        sum_squared += (imgs**2).sum(dim=(0, 2))

    mean = sum_channels / n_pixels
    std = (sum_squared / n_pixels - mean**2).sqrt()

    return mean.tolist(), std.tolist()
