import torch
from torch.utils.data import DataLoader, Dataset


def calculate_mean_and_std(dataset: Dataset, batch_size: int = 256) -> tuple[list[float], list[float]]:
    """
    Calculate per-channel mean and standard deviation for a dataset.

    This function iterates over the dataset in mini-batches to compute
    the mean and standard deviation for each channel in a memory-efficient manner.

    Args:
        dataset (Dataset): A PyTorch dataset that returns image tensors of shape [C, H, W].
        batch_size (int, optional): Number of samples per batch. Defaults to 256.

    Returns:
        tuple[list[float], list[float]]: A tuple containing two lists:
            - Mean values per channel.
            - Standard deviation values per channel.
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
