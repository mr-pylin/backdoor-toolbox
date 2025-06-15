from typing import Any, Callable, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2


class Subset(Subset):
    """
    A subset of a dataset with additional attributes for labels and metadata.

    Attributes:
        targets (Any): The labels of the subset.
        classes (list[str]): The list of class names.
        class_to_idx (dict[str, int]): A mapping from class names to indices.
        data (Any): The subset of data samples.
    """

    def __init__(self, dataset: Dataset, indices: NDArray[np.int32]):
        """
        Initialize a subset of the given dataset using specified indices.

        Args:
            dataset (Dataset): The parent dataset.
            indices (NDArray[np.int32]): Indices of samples to include.
        """
        super().__init__(dataset, indices)

        # store the dataset attributes
        self.targets = dataset.targets[indices]
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.data = dataset.data[indices]


class DatasetSplitter:
    """Splits a dataset into multiple subsets, optionally with overlapping samples."""

    def __init__(
        self,
        dataset: Dataset,
        num_subsets: int,
        subset_ratio: float = 0.0,
        overlap: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the DatasetSplitter.

        Args:
            dataset (Dataset): The dataset to split.
            num_subsets (int): Number of subsets to create.
            subset_ratio (float, optional): Ratio of data per subset. Defaults to 0.0.
            overlap (bool, optional): Whether subsets can share data. Defaults to False.
            seed (int, optional): Seed for reproducibility. Defaults to 42.
        """
        self.dataset = dataset
        self.num_subsets = num_subsets
        self.subset_ratio = subset_ratio
        self.overlap = overlap
        self.seed = seed

    def create_subsets(self) -> list[Subset]:
        """
        Create dataset subsets based on the specified configuration.

        Returns:
            list[Subset]: A list of generated subsets.
        """
        total_samples = len(self.dataset)
        indices = np.arange(total_samples)
        subset_size = int(total_samples * self.subset_ratio)
        subsets = []

        rng = np.random.default_rng(self.seed)

        if self.overlap:
            # overlapping subsets: Randomly sample with different seeds
            for model_id in range(self.num_subsets):
                model_rng = np.random.default_rng(self.seed + model_id)
                subset_indices = model_rng.choice(indices, size=subset_size, replace=False)
                subsets.append(Subset(self.dataset, subset_indices))
        else:
            # non-overlapping subsets: Partition the dataset
            rng.shuffle(indices)
            indices_per_model = np.array_split(indices, self.num_subsets)
            for model_indices in indices_per_model:
                subsets.append(Subset(self.dataset, model_indices))

        return subsets


# when skip_target_samples=True, there is a bug for added attributes in Subset
class PoisonedDatasetWrapper(Dataset):
    """A dataset wrapper that applies poisoning transformations to victim samples."""

    def __init__(
        self,
        base_dataset: Dataset,
        clean_transform: Optional[Callable] = None,
        clean_target_transform: Optional[Callable] = None,
        poison_transform: Optional[Callable] = None,
        poison_target_transform: Optional[Callable] = None,
        target_index: int = 0,
        victim_indices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        poison_ratio: float = 0.05,
        skip_target_samples: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the poisoned dataset wrapper.

        Args:
            base_dataset (Dataset): The dataset to wrap.
            clean_transform (Optional[Callable], optional): Transform for clean inputs.
            clean_target_transform (Optional[Callable], optional): Transform for clean labels.
            poison_transform (Optional[Callable], optional): Transform for poisoned inputs.
            poison_target_transform (Optional[Callable], optional): Transform for poisoned labels.
            target_index (int, optional): Poison label. Defaults to 0.
            victim_indices (tuple[int, ...], optional): Victim class labels. Defaults to digits 1–9.
            poison_ratio (float, optional): Fraction of victim samples to poison. Defaults to 0.05.
            skip_target_samples (bool, optional): Return only victims if True. Defaults to False.
            seed (int, optional): Random seed. Defaults to 42.
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

        # get indices of victim samples from the base dataset (or its subset)
        victim_mask = torch.isin(self.base_dataset.targets, torch.tensor(self.victim_indices))
        self.victim_samples_index = torch.nonzero(victim_mask, as_tuple=True)[0]

        # shuffle and select a subset for poisoning
        self.num_poison = int(len(self.victim_samples_index) * self.poison_ratio)
        self.poisoned_samples_index = self.victim_samples_index[
            torch.randperm(len(self.victim_samples_index))[: self.num_poison]
        ]

        # add necessary attributes from the base dataset
        self.targets = self.base_dataset.targets
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.data = self.base_dataset.data

    def __len__(self) -> int:
        """
        Return the number of accessible samples.

        Returns:
            int: Sample count (adjusted if `skip_target_samples` is True).
        """
        if self.skip_target_samples:
            return len(self.victim_samples_index)
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any, bool, Any]:
        """
        Get a transformed sample and metadata.

        Args:
            index (int): The index of the sample.

        Returns:
            tuple[Any, Any, bool, Any]: (image, label, poison_mask, raw_label)
        """

        # if we're only returning victim samples, adjust index.
        if self.skip_target_samples:
            index = self.victim_samples_index[index]

        image, label = self.base_dataset[index]

        # determine poison status based on the index in the original base dataset.
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
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

    from backdoor_toolbox.datasets.mnist import MNIST
    from backdoor_toolbox.triggers.target_transform import LabelFlip
    from backdoor_toolbox.triggers.transform.transform import InjectSolidTrigger
    from backdoor_toolbox.utils.dataset import DatasetSplitter, PoisonedDatasetWrapper

    # parameters
    root = "./data"
    target_index = 0
    victim_indices = tuple(range(1, 10))
    seed = 42
    nrows, ncols = 7, 20
    batch_size = nrows * ncols

    base_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]
    )

    poison_transform = v2.Compose(
        [
            InjectSolidTrigger(image_shape=(1, 28, 28), color=(1.0,), size=(6, 6), position=(20, 20)),
        ]
    )
    poison_target_transform = v2.Compose([LabelFlip(target_index=target_index)])

    # load base dataset
    mnist = MNIST(
        root=root,
        train=True,
        download=False,
        transform=base_transform,
        target_transform=None,
    )

    # create clean subsets
    subsets = DatasetSplitter(
        dataset=mnist,
        num_subsets=7,
        subset_ratio=0.5,
        overlap=False,
        seed=seed,
    ).create_subsets()

    # poisoned datasets
    poisoned_mnist_1 = PoisonedDatasetWrapper(
        base_dataset=subsets[0],
        clean_transform=None,
        clean_target_transform=None,
        poison_transform=poison_transform,
        poison_target_transform=poison_target_transform,
        target_index=target_index,
        victim_indices=victim_indices,
        poison_ratio=0.05,
        skip_target_samples=False,
        seed=seed,
    )

    poisoned_mnist_2 = PoisonedDatasetWrapper(
        base_dataset=subsets[1],
        clean_transform=None,
        clean_target_transform=None,
        poison_transform=poison_transform,
        poison_target_transform=poison_target_transform,
        target_index=target_index,
        victim_indices=victim_indices,
        poison_ratio=1.0,
        skip_target_samples=True,
        seed=seed,
    )

    # log subset info
    for i, subset in enumerate(subsets):
        print(f"subsets[{i}]:")
        print(f"\tlen          : {len(subset)}")
        print(f"\ttargets.dtype: {subset.targets.dtype}")
        print(f"\ttargets.shape: {subset.targets.shape}\n")

    for name, ds in zip(["poisoned_mnist_1", "poisoned_mnist_2"], [poisoned_mnist_1, poisoned_mnist_2]):
        print(f"{name}:")
        print(f"\tlen          : {len(ds)}")
        print(f"\ttargets.dtype: {ds.targets.dtype}")
        print(f"\ttargets.shape: {ds.targets.shape}\n")

    # load batches
    poisoned_loader_1 = next(iter(DataLoader(poisoned_mnist_1, batch_size=batch_size, shuffle=False)))
    poisoned_loader_2 = next(iter(DataLoader(poisoned_mnist_2, batch_size=batch_size, shuffle=False)))
    clean_loader = next(iter(DataLoader(subsets[2], batch_size=batch_size, shuffle=False)))

    # plot poisoned_mnist_1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols, nrows), layout="compressed")
    plt.suptitle("First batch from poisoned_mnist_1")
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            axs[i, j].imshow(poisoned_loader_1[0][idx].permute(1, 2, 0), cmap="gray")
            axs[i, j].set_title(
                f"{poisoned_loader_1[1][idx].item()}, {poisoned_loader_1[3][idx].item()}, {str(poisoned_loader_1[2][idx].item())[0]}"
            )
            axs[i, j].axis("off")

    # plot poisoned_mnist_2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols, nrows), layout="compressed")
    plt.suptitle("First batch from poisoned_mnist_2")
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            axs[i, j].imshow(poisoned_loader_2[0][idx].permute(1, 2, 0), cmap="gray")
            axs[i, j].set_title(
                f"{poisoned_loader_2[1][idx].item()}, {poisoned_loader_2[3][idx].item()}, {str(poisoned_loader_2[2][idx].item())[0]}"
            )
            axs[i, j].axis("off")

    # plot clean_mnist
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols, nrows), layout="compressed")
    plt.suptitle("First batch from clean_mnist")
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            axs[i, j].imshow(clean_loader[0][idx].permute(1, 2, 0), cmap="gray")
            axs[i, j].set_title(f"{clean_loader[1][idx].item()}")
            axs[i, j].axis("off")

    plt.show()
