from pathlib import Path

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2


class CleanMNIST(MNIST):
    """
    A clean version of the MNIST dataset with reproducible behavior and corrections to dataset attributes.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: v2.Compose | None = None,
        target_transform: v2.Compose | None = None,
        download: bool = False,
        seed: int | float = 42,
    ):
        """
        Initializes the CleanMNIST dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool): Whether to load the training set. Defaults to True.
            transform (v2.Compose | None): Transformations for images. Defaults to None.
            target_transform (v2.Compose | None): Transformations for labels. Defaults to None.
            download (bool): Whether to download the dataset if not found. Defaults to False.
            seed (int | float): Seed for reproducibility. Defaults to 42.
        """
        torch.manual_seed(seed)
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # check and convert `self.targets` to `torch.Tensor`
        if not isinstance(self.targets, torch.Tensor):
            self.targets = torch.tensor(self.targets, dtype=torch.int64)

    def __len__(self):
        """Returns the total number of samples."""
        return super().__len__()

    def __getitem__(self, index):
        """
        Retrieves an image and its label by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: The image tensor and its label.
        """
        image, label = super().__getitem__(index)
        return image, label

    @property
    def raw_folder(self) -> str:
        """
        The folder containing the raw dataset.

        Returns:
            str: Path to the raw dataset folder.
        """
        return Path(self.root) / self.__class__.__bases__[0].__name__ / "raw"


class PoisonedMNIST(MNIST):
    """
    A modified version of the MNIST dataset for backdoor attack experiments with poisoned samples.

    Attributes:
        clean_transform (v2.Compose | None): Transformations for clean images.
        clean_target_transform (v2.Compose | None): Transformations for clean labels.
        poisoned_transform (v2.Compose | None): Transformations for poisoned images.
        poisoned_target_transform (v2.Compose | None): Transformations for poisoned labels.
        victim_samples_index (torch.Tensor): Indices of victim samples.
        poisoned_samples_index (torch.Tensor): Indices of poisoned samples.
        num_poison (int): Number of poisoned samples.
        seed (int | float): Seed for reproducibility.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        clean_transform: v2.Compose | None = None,
        clean_target_transform: v2.Compose | None = None,
        download: bool = False,
        target_index: int = 0,
        victim_indices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        poison_ratio: float = 0.05,
        poisoned_transform: v2.Compose | None = None,
        poisoned_target_transform: v2.Compose | None = None,
        skip_target_samples: bool = False,
        seed: int | float = 42,
    ):
        """
        Initializes the PoisonedMNIST dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool): Whether to load the training set. Defaults to True.
            clean_transform (v2.Compose | None): Transformations for clean images. Defaults to None.
            clean_target_transform (v2.Compose | None): Transformations for clean labels. Defaults to None.
            download (bool): Whether to download the dataset if not found. Defaults to False.
            target_index (int): The target label for poisoned samples. Defaults to 0.
            victim_indices (Tuple[int, ...]): Labels eligible for poisoning. Defaults to (1, 2, 3, 4, 5, 6, 7, 8, 9).
            poison_ratio (float): Proportion of samples to poison. Defaults to 0.05.
            poisoned_transform (v2.Compose | None): Transformations for poisoned images. Defaults to None.
            poisoned_target_transform (v2.Compose | None): Transformations for poisoned labels. Defaults to None.
            skip_target_samples (bool): Whether to return only victim samples. Defaults to False.
            seed (int | float): Seed for reproducibility. Defaults to 42.
        """
        torch.manual_seed(seed)
        super().__init__(root, train=train, transform=None, target_transform=None, download=download)
        self.clean_transform = clean_transform
        self.clean_target_transform = clean_target_transform
        self.target_index = target_index
        self.victim_indices = victim_indices
        self.poison_ratio = poison_ratio
        self.poisoned_transform = poisoned_transform
        self.poisoned_target_transform = poisoned_target_transform
        self.skip_target_samples = skip_target_samples
        self.seed = seed

        # check and convert `self.targets` to `torch.Tensor`
        if not isinstance(self.targets, torch.Tensor):
            self.targets = torch.tensor(self.targets, dtype=torch.int64)

        # get indices of victim samples
        victim_mask = torch.isin(self.targets, torch.tensor(self.victim_indices))
        self.victim_samples_index = torch.nonzero(victim_mask, as_tuple=True)[0]

        # shuffle and select a subset for poisoning
        self.num_poison = int(len(self.victim_samples_index) * self.poison_ratio)
        self.poisoned_samples_index = self.victim_samples_index[torch.randperm(len(self.victim_samples_index))[: self.num_poison]]

        # store flipped targets for poisoned images
        # self.targets_poisoned = self.targets.clone()
        # self.targets_poisoned[self.poisoned_samples_index] = self.target_index

    def __len__(self):
        """
        Returns the total number of samples or only victim samples if `skip_target_samples` is True.

        Returns:
            int: Number of samples.
        """
        if self.skip_target_samples:
            return len(self.victim_samples_index)
        return super().__len__()

    def __getitem__(self, index):
        """
        Retrieves an image, label, poison status, and raw label by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int, bool, int]:
                - Image tensor.
                - Transformed label.
                - Whether the sample is poisoned.
                - Raw label before transformation.
        """
        if self.skip_target_samples:
            index = self.victim_samples_index[index]

        image, label = super().__getitem__(index)

        # determine if the sample is poisoned or clean
        is_poisoned = index in self.poisoned_samples_index
        raw_label = label

        # apply clean_transforms to clean samples and poisoned_transforms to poisoned samples
        if is_poisoned:
            poison_mask = True
            if self.poisoned_transform is not None:
                image = self.poisoned_transform(image)
            if self.poisoned_target_transform is not None:
                label = self.poisoned_target_transform(label)
        else:
            poison_mask = False
            if self.clean_transform is not None:
                image = self.clean_transform(image)
            if self.clean_target_transform is not None:
                label = self.clean_target_transform(label)

        return image, label, poison_mask, raw_label

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
    from torch.utils.data import DataLoader

    from backdoor_toolbox.triggers.target_transform import LabelFlip
    from backdoor_toolbox.triggers.transform import InjectSolidTrigger

    ROOT = "./data"
    TARGET_INDEX = 0
    VICTIM_INDICES = tuple(range(1, 10))
    SEED = 42
    CLEAN_TRANSFORM = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]
    )
    CLEAN_TARGET_TRANSFORM = None
    POISONED_TRANSFORM = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            InjectSolidTrigger(image_shape=(1, 28, 28), color=(1.0,), size=(6, 6), position=(20, 20)),
        ]
    )
    POISONED_TARGET_TRANSFORM = v2.Compose(
        [
            LabelFlip(target_index=TARGET_INDEX),
        ]
    )

    # create a poisoned MNIST dataset
    poisoned_mnist_1 = PoisonedMNIST(
        root=ROOT,
        train=True,
        clean_transform=CLEAN_TRANSFORM,
        clean_target_transform=CLEAN_TARGET_TRANSFORM,
        download=False,
        target_index=TARGET_INDEX,
        victim_indices=VICTIM_INDICES,
        poison_ratio=0.05,
        poisoned_transform=POISONED_TRANSFORM,
        poisoned_target_transform=POISONED_TARGET_TRANSFORM,
        skip_target_samples=False,
        seed=SEED,
    )

    # create another poisoned MNIST dataset
    poisoned_mnist_2 = PoisonedMNIST(
        root=ROOT,
        train=True,
        clean_transform=CLEAN_TRANSFORM,
        clean_target_transform=CLEAN_TARGET_TRANSFORM,
        download=False,
        target_index=TARGET_INDEX,
        victim_indices=VICTIM_INDICES,
        poison_ratio=1.0,
        poisoned_transform=POISONED_TRANSFORM,
        poisoned_target_transform=POISONED_TARGET_TRANSFORM,
        skip_target_samples=True,
        seed=SEED,
    )

    # create a clean MNIST dataset
    clean_mnist = CleanMNIST(
        root=ROOT,
        train=True,
        transform=CLEAN_TRANSFORM,
        target_transform=CLEAN_TARGET_TRANSFORM,
        download=False,
        seed=SEED,
    )

    # log
    print(f"poisoned_mnist_1:")
    print(f"\tlen(poisoned_mnist_1)          : {len(poisoned_mnist_1)}")
    print(f"\tpoisoned_mnist_1.targets.dtype : {poisoned_mnist_1.targets.dtype}")
    print(f"\tpoisoned_mnist_1.targets.shape : {poisoned_mnist_1.targets.shape}\n")
    print(f"poisoned_mnist_2:")
    print(f"\tlen(poisoned_mnist_2)          : {len(poisoned_mnist_2)}")
    print(f"\tpoisoned_mnist_2.targets.dtype : {poisoned_mnist_2.targets.dtype}")
    print(f"\tpoisoned_mnist_2.targets.shape : {poisoned_mnist_2.targets.shape}\n")
    print(f"clean_mnist:")
    print(f"\tlen(clean_mnist)          : {len(clean_mnist)}")
    print(f"\tclean_mnist.targets.dtype : {clean_mnist.targets.dtype}")
    print(f"\tclean_mnist.targets.shape : {clean_mnist.targets.shape}")

    # create data loaders and fetch the first batch
    NROWS, NCOLS = 7, 20
    poisoned_mnist_dl_1 = next(iter(DataLoader(poisoned_mnist_1, batch_size=NROWS * NCOLS, shuffle=False)))
    poisoned_mnist_dl_2 = next(iter(DataLoader(poisoned_mnist_2, batch_size=NROWS * NCOLS, shuffle=False)))
    clean_mnist_dl = next(iter(DataLoader(clean_mnist, batch_size=NROWS * NCOLS, shuffle=False)))

    # plot
    fig, axs = plt.subplots(NROWS, NCOLS, figsize=(NCOLS, NROWS), layout="compressed")
    plt.suptitle(f"First {NROWS * NCOLS} images of poisoned_mnist_1")
    for i in range(NROWS):
        for j in range(NCOLS):
            axs[i, j].imshow(poisoned_mnist_dl_1[0][i * NCOLS + j].permute(1, 2, 0), cmap="gray")
            axs[i, j].set_title(
                f"{poisoned_mnist_dl_1[1][i * NCOLS + j].item()}, {poisoned_mnist_dl_1[3][i * NCOLS + j].item()}, {str(poisoned_mnist_dl_1[2][i * NCOLS + j].item())[0]}"
            )
            axs[i, j].axis("off")

    fig, axs = plt.subplots(NROWS, NCOLS, figsize=(NCOLS, NROWS), layout="compressed")
    plt.suptitle(f"First {NROWS * NCOLS} images of poisoned_mnist_2")
    for i in range(NROWS):
        for j in range(NCOLS):
            axs[i, j].imshow(poisoned_mnist_dl_2[0][i * NCOLS + j].permute(1, 2, 0), cmap="gray")
            axs[i, j].set_title(
                f"{poisoned_mnist_dl_2[1][i * NCOLS + j].item()}, {poisoned_mnist_dl_2[3][i * NCOLS + j].item()}, {str(poisoned_mnist_dl_2[2][i * NCOLS + j].item())[0]}"
            )
            axs[i, j].axis("off")

    fig, axs = plt.subplots(NROWS, NCOLS, figsize=(NCOLS, NROWS), layout="compressed")
    plt.suptitle(f"First {NROWS * NCOLS} images of clean_mnist")
    for i in range(NROWS):
        for j in range(NCOLS):
            axs[i, j].imshow(clean_mnist_dl[0][i * NCOLS + j].permute(1, 2, 0), cmap="gray")
            axs[i, j].set_title(clean_mnist_dl[1][i * NCOLS + j].item())
            axs[i, j].axis("off")

    plt.show()
