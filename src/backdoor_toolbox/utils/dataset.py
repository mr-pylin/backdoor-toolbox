from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, GTSRB, MNIST
from torchvision.transforms import v2


class Subset(Subset):
    """
    A subset of a dataset with additional attributes.

    Attributes:
        targets (List[int]): The labels of the subset.
        classes (List[str]): The list of class names.
        class_to_idx (Dict[str, int]): A mapping from class names to indices.
        data (Any): The subset of data samples.
    """

    def __init__(self, dataset: Dataset, indices: np.ndarray[int]):
        """
        Initializes the Subset object by selecting the specified indices from the dataset.

        Args:
            dataset (Dataset): The parent dataset from which the subset is created.
            indices (np.ndarray[int]): The indices of the samples to include in the subset.
        """
        super().__init__(dataset, indices)

        # store the dataset attributes
        self.targets = dataset.targets[indices]
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.data = dataset.data[indices]


class DatasetSplitter:
    """
    Splits a dataset into multiple subsets with optional overlap.

    Attributes:
        dataset (torch.utils.data.Dataset): The dataset to split.
        num_subsets (int): The number of subsets to create.
        subset_ratio (float): The ratio of the dataset to include in each subset.
        overlap (bool): Whether subsets can have overlapping samples.
        seed (int): The random seed for reproducibility.
        subsets (List[Subset]): The list of created subsets.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_subsets: int,
        subset_ratio: float = 0.0,
        overlap: bool = False,
        seed: int = 42,
    ):
        """
        Initializes the DatasetSplitter with the given parameters.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to split.
            num_subsets (int): The number of subsets to create.
            subset_ratio (float, optional): The ratio of the dataset to include in each subset. Defaults to 0.0.
            overlap (bool, optional): Whether subsets can have overlapping samples. Defaults to False.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
        """
        self.dataset = dataset
        self.num_subsets = num_subsets
        self.subset_ratio = subset_ratio
        self.overlap = overlap
        self.seed = seed

    def create_subsets(self) -> list[Subset]:
        """
        Creates subsets from the dataset based on the specified parameters.

        Returns:
            list[Subset]: A list of created subsets.
        """
        total_samples = len(self.dataset)
        indices = np.arange(total_samples)
        subset_size = int(total_samples * self.subset_ratio)
        subsets = []

        rng = np.random.default_rng(self.seed)

        if self.overlap:
            # Overlapping subsets: Randomly sample with different seeds
            for model_id in range(self.num_subsets):
                model_rng = np.random.default_rng(self.seed + model_id)
                subset_indices = model_rng.choice(indices, size=subset_size, replace=False)
                subsets.append(Subset(self.dataset, subset_indices))
        else:
            # Non-overlapping subsets: Partition the dataset
            rng.shuffle(indices)
            indices_per_model = np.array_split(indices, self.num_subsets)
            for model_indices in indices_per_model:
                subsets.append(Subset(self.dataset, model_indices))

        return subsets


# when skip_target_samples=True, there is a bug for added attributes in Subset
class PoisonedDatasetWrapper(Dataset):
    """
    A wrapper that applies backdoor poisoning logic to an existing dataset.

    Attributes:
        base_dataset (Dataset): The original dataset to be wrapped.
        clean_transform (Callable, optional): Transform to apply to clean images.
        clean_target_transform (Callable, optional): Transform to apply to clean labels.
        poison_transform (Callable, optional): Transform to apply to poisoned images.
        poison_target_transform (Callable, optional): Transform to apply to poisoned labels.
        target_index (int): The target class index for poisoning.
        victim_indices (Tuple[int, ...]): The class indices to be targeted.
        poison_ratio (float): The ratio of victim samples to poison.
        skip_target_samples (bool): Whether to skip non-victim samples.
        seed (int): The random seed for reproducibility.
        victim_samples_index (torch.Tensor): Indices of victim samples.
        poisoned_samples_index (torch.Tensor): Indices of poisoned victim samples.
        targets (Any): The labels of the base dataset.
        classes (List[str]): The list of class names.
        class_to_idx (Dict[str, int]): A mapping from class names to indices.
        data (Any): The dataset samples.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        clean_transform=None,
        clean_target_transform=None,
        poison_transform=None,
        poison_target_transform=None,
        target_index: int = 0,
        victim_indices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        poison_ratio: float = 0.05,
        skip_target_samples: bool = False,
        seed: int = 42,
    ):
        """
        Initializes the PoisonedDatasetWrapper with the given parameters.

        Args:
            base_dataset (Dataset): The original dataset to be wrapped.
            clean_transform (Callable, optional): Transform to apply to clean images. Defaults to None.
            clean_target_transform (Callable, optional): Transform to apply to clean labels. Defaults to None.
            poison_transform (Callable, optional): Transform to apply to poisoned images. Defaults to None.
            poison_target_transform (Callable, optional): Transform to apply to poisoned labels. Defaults to None.
            target_index (int, optional): The target class index for poisoning. Defaults to 0.
            victim_indices (Tuple[int, ...], optional): The class indices to be targeted. Defaults to (1-9).
            poison_ratio (float, optional): The ratio of victim samples to poison. Defaults to 0.05.
            skip_target_samples (bool, optional): Whether to skip non-victim samples. Defaults to False.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
        """
        self.base_dataset = base_dataset
        self.clean_transform = clean_transform
        self.clean_target_transform = clean_target_transform
        self.target_index = target_index
        self.victim_indices = victim_indices
        self.poison_ratio = poison_ratio
        self.poison_transform = poison_transform
        self.poison_target_transform = poison_target_transform
        self.skip_target_samples = skip_target_samples
        self.seed = seed

        torch.manual_seed(seed)

        # Get indices of victim samples from the base dataset (or its subset)
        victim_mask = torch.isin(self.base_dataset.targets, torch.tensor(self.victim_indices))
        self.victim_samples_index = torch.nonzero(victim_mask, as_tuple=True)[0]

        # Shuffle and select a subset for poisoning
        self.num_poison = int(len(self.victim_samples_index) * self.poison_ratio)
        self.poisoned_samples_index = self.victim_samples_index[
            torch.randperm(len(self.victim_samples_index))[: self.num_poison]
        ]

        # Add necessary attributes from the base dataset
        self.targets = self.base_dataset.targets
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.data = self.base_dataset.data

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        if self.skip_target_samples:
            return len(self.victim_samples_index)
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Retrieves a sample and its associated metadata.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple: (image, label, poison_mask, raw_label)
        """
        # If we're only returning victim samples, adjust index.
        if self.skip_target_samples:
            index = self.victim_samples_index[index]

        image, label = self.base_dataset[index]

        # Determine poison status based on the index in the original base dataset.
        # Note: If using a Subset, index here is relative to the subset.
        is_poisoned = index in self.poisoned_samples_index
        raw_label = label

        if is_poisoned:
            if self.poison_transform is not None:
                image = self.poison_transform(image)
            if self.poison_target_transform is not None:
                label = self.poison_target_transform(label)
            poison_mask = True
        else:
            if self.clean_transform is not None:
                image = self.clean_transform(image)
            if self.clean_target_transform is not None:
                label = self.clean_target_transform(label)
            poison_mask = False

        return image, label, poison_mask, raw_label


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from backdoor_toolbox.datasets import MNIST
    from backdoor_toolbox.triggers.target_transform import LabelFlip
    from backdoor_toolbox.triggers.transform.transform import InjectSolidTrigger

    # define parameters
    ROOT = "./data"
    TARGET_INDEX = 0
    VICTIM_INDICES = tuple(range(1, 10))
    SEED = 42
    BASE_TRANSFORM = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]
    )
    BASE_TARGET_TRANSFORM = None
    poison_transform = v2.Compose(
        [
            # v2.ToImage(),
            # v2.ToDtype(dtype=torch.float32, scale=True),
            InjectSolidTrigger(image_shape=(1, 28, 28), color=(1.0,), size=(6, 6), position=(20, 20)),
        ]
    )
    poison_target_transform = v2.Compose([LabelFlip(target_index=TARGET_INDEX)])

    # initialize dataset
    mnist = MNIST(
        root=ROOT,
        train=True,
        download=False,
        transform=BASE_TRANSFORM,
        target_transform=BASE_TARGET_TRANSFORM,
    )

    # initialize subsets
    subsets = DatasetSplitter(
        dataset=mnist,
        num_subsets=7,
        subset_ratio=0.5,
        overlap=False,
        seed=42,
    ).get_subsets()

    # create a poisoned MNIST dataset
    poisoned_mnist_1 = PoisonedDatasetWrapper(
        base_dataset=subsets[0],
        clean_transform=None,
        clean_target_transform=None,
        poison_transform=poison_transform,
        poison_target_transform=poison_target_transform,
        target_index=TARGET_INDEX,
        victim_indices=VICTIM_INDICES,
        poison_ratio=0.05,
        skip_target_samples=False,
        seed=SEED,
    )

    # create another poisoned MNIST dataset
    poisoned_mnist_2 = PoisonedDatasetWrapper(
        base_dataset=subsets[1],
        clean_transform=None,
        clean_target_transform=None,
        poison_transform=poison_transform,
        poison_target_transform=poison_target_transform,
        target_index=TARGET_INDEX,
        victim_indices=VICTIM_INDICES,
        poison_ratio=1.0,
        skip_target_samples=True,
        seed=SEED,
    )

    # log
    # print(subsets[0].dataset.transform)
    # print(poisoned_mnist_1.base_dataset.dataset.transform)
    # print(poisoned_mnist_1[0])

    for i in range(len(subsets)):
        print(f"subsets[{i}]:")
        print(f"\tlen(subsets[{i}])          : {len(subsets[i])}")
        print(f"\tsubsets[{i}].targets.dtype : {subsets[i].targets.dtype}")
        print(f"\tsubsets[{i}].targets.shape : {subsets[i].targets.shape}\n")

    print(f"poisoned_mnist_1:")
    print(f"\tlen(poisoned_mnist_1)          : {len(poisoned_mnist_1)}")
    print(f"\tpoisoned_mnist_1.targets.dtype : {poisoned_mnist_1.targets.dtype}")
    print(f"\tpoisoned_mnist_1.targets.shape : {poisoned_mnist_1.targets.shape}\n")

    print(f"poisoned_mnist_2:")
    print(f"\tlen(poisoned_mnist_2)          : {len(poisoned_mnist_2)}")
    print(f"\tpoisoned_mnist_2.targets.dtype : {poisoned_mnist_2.targets.dtype}")
    print(f"\tpoisoned_mnist_2.targets.shape : {poisoned_mnist_2.targets.shape}\n")

    # create data loaders and fetch the first batch
    NROWS, NCOLS = 7, 20
    poisoned_mnist_dl_1 = next(iter(DataLoader(poisoned_mnist_1, batch_size=NROWS * NCOLS, shuffle=False)))
    poisoned_mnist_dl_2 = next(iter(DataLoader(poisoned_mnist_2, batch_size=NROWS * NCOLS, shuffle=False)))
    clean_mnist_dl = next(iter(DataLoader(subsets[2], batch_size=NROWS * NCOLS, shuffle=False)))

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
