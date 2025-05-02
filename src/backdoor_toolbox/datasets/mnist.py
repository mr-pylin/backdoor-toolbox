from pathlib import Path

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2


class MNIST(MNIST):
    """
    A clean version of the MNIST dataset with reproducible behavior and corrections to dataset attributes.
    Inherits from `torchvision.datasets.MNIST`.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: v2.Transform | None = None,
        target_transform: v2.Transform | None = None,
    ):
        """
        Initializes the MNIST dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool, optional): Whether to load the training set. Defaults to True.
            download (bool, optional): Whether to download the dataset if not found. Defaults to False.
            transform (v2.Transform | None, optional): Transformations for images. Defaults to None.
            target_transform (v2.Transform | None, optional): Transformations for labels. Defaults to None.
        """
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return super().__len__()

    def __getitem__(self, index):
        """
        Retrieves an image and its label by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the image tensor and its label.
        """
        image, label = super().__getitem__(index)
        label = torch.tensor(label, dtype=torch.int64)
        return image, label

    @property
    def raw_folder(self) -> str:
        """
        The folder containing the raw dataset.

        Returns:
            str: Path to the raw dataset folder.
        """
        return Path(self.root) / self.__class__.__bases__[0].__name__ / "raw"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # define parameters
    root = "./data"
    train = True
    download = True

    # initialize dataset
    dataset = MNIST(root=root, train=train, download=download)

    # basic tests
    print(f"Dataset size      : {len(dataset)}")
    print(f"Classes           : {dataset.classes if hasattr(dataset, 'classes') else 'Not defined'}")
    print(f"First sample size : {dataset[0][0].size}")
    print(f"First label       : {dataset[0][1]}")

    # check for GPU compatibility (optional)
    image, label = v2.ToImage()(dataset[0][0]), dataset[0][1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving sample to {device}")
    image = image.to(device)
    print(f"Sample tensor device: {image.device}")

    # plot first 140 images in 7 rows and 20 columns
    fig, axes = plt.subplots(7, 20, figsize=(20, 7), layout="compressed")
    for i, ax in enumerate(axes.flatten()):
        img, label = dataset[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(str(label), fontsize=8)
        ax.axis("off")
    plt.show()
