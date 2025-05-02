import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


class CIFAR10(CIFAR10):
    """
    A clean version of the CIFAR10 dataset with reproducible behavior and corrections to dataset attributes.
    Inherits from `torchvision.datasets.CIFAR10`.
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
        Initializes the CIFAR10 dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool, optional): Whether to load the training set. Defaults to True.
            download (bool, optional): Whether to download the dataset if not found. Defaults to False.
            transform (v2.Transform | None, optional): Transformations for images. Defaults to None.
            target_transform (v2.Transform | None, optional): Transformations for labels. Defaults to None.
        """
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.targets = torch.tensor(self.targets, dtype=torch.int64)

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
        return image, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # define parameters
    root = "./data"
    train = True
    download = True

    # initialize dataset
    dataset = CIFAR10(root=root, train=train, download=download)

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

    print(img.size)
