from abc import ABC, abstractmethod
import torch


class InjectTrigger(ABC):
    """
    Abstract base class for injecting triggers into images in backdoor attacks.

    Attributes:
        image_shape (tuple[int, int, int]): Shape of the input image (C, H, W).
    """

    def __init__(self, image_shape: tuple[int, int, int]):
        """
        Initializes the InjectTrigger class.

        Args:
            image_shape (tuple[int, int, int]): Shape of the input image (C, H, W).
        """
        super().__init__()
        self.image_shape = image_shape

    @abstractmethod
    def _generate_mask(self) -> torch.Tensor:
        """
        Abstract method to generate a binary mask for the trigger region.

        Args:
            image_shape (tuple[int, int, int]): Shape of the input image (C, H, W).

        Returns:
            torch.Tensor: Binary mask of the same shape as the image.
        """
        pass

    @abstractmethod
    def _generate_trigger(self) -> torch.Tensor:
        """
        Abstract method to generate the trigger pattern.

        Args:
            image_shape (tuple[int, int, int]): Shape of the input image (C, H, W).

        Returns:
            torch.Tensor: Trigger pattern of the same shape as the image.
        """
        pass

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the trigger to the input image.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Image tensor with the trigger applied.
        """
        mask = self._generate_mask(self.image_shape)
        trigger = self._generate_trigger(self.image_shape)
        return (1 - mask) * image + mask * trigger
