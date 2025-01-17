from enum import Enum

import numpy as np
import torch
from torchvision.transforms import v2


class LabelFlip:
    """
    A transformation class for flipping labels to a specified target index.

    Attributes:
        target_index (int | None): The target label to which all input labels will be flipped.
                                   If `None`, labels remain unchanged.
    """

    def __init__(self, target_index: int | None = None):
        """
        Initializes the LabelFlip transformation.

        Args:
            target_index (int | None): The target label for flipping. If `None`, no flipping is applied.
        """
        self.target_index = target_index

    def __call__(self, labels: int | list | np.ndarray | torch.Tensor) -> int | list | np.ndarray | torch.Tensor:
        """
        Flips the input labels to the target label.

        Args:
            labels (int | list[int] | np.ndarray | torch.Tensor): The input labels to transform.

        Returns:
            int | list[int] | np.ndarray | torch.Tensor: The transformed labels.
                - If `target_index` is set, all labels are flipped to the specified value.
                - Otherwise, the original labels are returned.
        """
        if self.target_index is not None:
            if isinstance(labels, torch.Tensor):
                return torch.full_like(labels, self.target_index)
            if isinstance(labels, np.ndarray):
                return np.full_like(labels, self.target_index)
            elif isinstance(labels, list):
                return [self.target_index] * len(labels)
            elif isinstance(labels, int):
                return self.target_index
        return labels

    def __repr__(self):
        """
        Returns a string representation of the LabelFlip instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"{self.__class__.__name__}(target_label={self.target_index})"


class TargetTriggerTypes(Enum):
    FLIPLABEL = LabelFlip


if __name__ == "__main__":

    examples_of_targets = {
        "tensor": torch.tensor([1, 2, 3, 0, 2, 0, 1, 2]),
        "numpy": np.array([1, 2, 3, 0, 2, 0, 1, 2]),
        "list": [1, 2, 3, 0, 2, 0, 1, 2],
        "int": 2,
    }

    for t, targets in examples_of_targets.items():
        print(f"targets : {targets}")

        # case 1: `target_label = None`
        target_index = None
        target_transform = v2.Compose([TargetTriggerTypes.FLIPLABEL.value(target_index)])
        transformed_targets = target_transform(targets)
        print(f"\tcase 1:")
        print(f"\t\ttarget_index              : {target_index}")
        print(f"\t\ttarget_transform          : {target_transform.transforms[0]}")
        print(f"\t\ttransformed_targets       : {transformed_targets}")
        print(f"\t\ttype(transformed_targets) : {type(transformed_targets)}\n")

        # case 2: `target_label = 0`
        target_index = 0
        target_transform = v2.Compose([TargetTriggerTypes.FLIPLABEL.value(target_index)])
        transformed_targets = target_transform(targets)
        print(f"\tcase 2:")
        print(f"\t\ttarget_index              : {target_index}")
        print(f"\t\ttarget_transform          : {target_transform.transforms[0]}")
        print(f"\t\ttransformed_targets       : {transformed_targets}")
        print(f"\t\ttype(transformed_targets) : {type(transformed_targets)}\n")
