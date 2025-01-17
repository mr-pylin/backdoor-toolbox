import importlib
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BaseRoutine(ABC):
    """
    Abstract base class for routines that handle the process of data preparation, model preparation, and application of attacks or defenses.

    Methods:
        apply: Applies the routine (to be implemented in subclasses).
        _prepare_data: Prepares the dataset and dataloaders for training/testing.
        _prepare_model: Prepares the model with specified weights.
        _import_package: Imports a package dynamically.
        _calculate_train_set_mean_and_std: Calculates the mean and standard deviation for the training dataset.
    """

    @abstractmethod
    def apply(self):
        """
        Applies the routine (to be implemented in subclasses).
        """
        pass

    @abstractmethod
    def _prepare_data(
        self,
        module_path: str,
    ) -> tuple[tuple[Dataset, ...], tuple[DataLoader, ...]]:
        """
        Prepares the dataset and dataloaders for training/testing.

        Args:
            module_path (str): The path to the module where the dataset resides.

        Returns:
            Tuple[Tuple[Dataset, ...], Tuple[DataLoader, ...]]: A tuple containing datasets and dataloaders.
        """
        pass

    @abstractmethod
    def _prepare_model(
        self,
        module_path: str,
        module_cls: str,
        weights: bool | str,
    ) -> nn.Module:
        """
        Prepares the model with specified weights.

        Args:
            module_path (str): The path to the module containing the model.
            module_cls (str): The class name of the model.
            weights (Union[bool, str]): Weights to load for the model (either a boolean indicating if pretrained weights should be loaded, or a path to a weights file).

        Returns:
            nn.Module: The prepared model.
        """
        pass

    def _import_package(self, package: str) -> object:
        """
        Dynamically imports a package.

        Args:
            package (str): The name of the package to import.

        Returns:
            object: The imported module.

        Raises:
            ImportError: If the package cannot be imported.
        """
        try:
            module = importlib.import_module(package)
            return module
        except ModuleNotFoundError as e:
            raise ImportError(f"Error importing package '{package}': {e}")

    def _calculate_train_set_mean_and_std(self, train_set: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and standard deviation of the training dataset.

        Args:
            train_set (Dataset): The training dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation per channel.
        """
        load_entire_train_set_samples: torch.Tensor = next(iter(DataLoader(dataset=train_set, batch_size=len(train_set), shuffle=False)))[0]
        mean_per_channel = load_entire_train_set_samples.mean(dim=(0, 2, 3)).tolist()
        std_per_channel = load_entire_train_set_samples.std(dim=(0, 2, 3)).tolist()

        return mean_per_channel, std_per_channel
