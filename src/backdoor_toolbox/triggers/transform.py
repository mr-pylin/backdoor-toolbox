from abc import ABC, abstractmethod
from enum import Enum

import torch
from torchvision.transforms import v2


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


class InjectSolidTrigger(InjectTrigger):
    """
    Class for injecting a solid-colored trigger into an image.

    Attributes:
        color (torch.Tensor): Color of the trigger (tuple of floats, one for each channel).
        size (tuple[int, int]): Size of the trigger (width, height).
        position (tuple[int, int]): Position of the top-left corner of the trigger (x, y).
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        color: tuple[float, ...] = (1.0,),
        size: tuple[int, int] = (6, 6),
        position: tuple[int, int] = (0, 0),
    ):
        """
        Initializes the InjectSolidTrigger class.

        Args:
            image_shape (tuple[int, int, int]): Shape of the input image (C, H, W).
            color (tuple[float, ...]): Color of the trigger, one value per channel. Defaults to (1.0,).
            size (tuple[int, int]): Size of the trigger (width, height). Defaults to (6, 6).
            position (tuple[int, int]): Position of the trigger (x, y). Defaults to (0, 0).

        Raises:
            ValueError: If the color, size, or position parameters are invalid.
        """
        super().__init__(image_shape)
        self.color = torch.tensor(color, dtype=torch.float32)
        self.size = size
        self.position = position

        # validate parameters
        if len(self.color) < 1:
            print(isinstance(self.color, tuple))
            print(len(self.color) > 0)
            raise ValueError("Trigger color must be a `tuple` of at-least 1 element e.g. (1.0,) or (1.0, 1.0, 1.0).")
        if len(self.size) != 2:
            raise ValueError("Trigger size must be a `tuple` of 2 elements (Row, Column)[depth is inferred].")
        if len(self.position) != 2:
            raise ValueError("Position must be a tuple of 2 elements (x, y) [depth is inferred].")

    def _generate_mask(self, image_shape: tuple[int, int, int]) -> torch.Tensor:
        """
        Generates a binary mask for the solid trigger region.

        Args:
            image_shape (tuple[int, int, int]): Shape of the input image (C, H, W).

        Returns:
            torch.Tensor: Binary mask of the same shape as the input image.

        Raises:
            ValueError: If the input image shape is invalid.
        """
        if len(image_shape) != 3:
            raise ValueError(f"Input image must have 3 dimensions (C, H, W), but got {len(image_shape)} dimensions.")
        if image_shape[0] != len(self.color):
            raise ValueError(f"Number of channels of `image_shape`:{image_shape[0]} is not equal to length of `self.color`:{len(self.color)}")

        C, H, W = image_shape
        mask = torch.zeros((C, H, W))

        # get the position and size of the trigger
        pos_x, pos_y = self.position
        size_x, size_y = self.size

        # set mask to 1 in the trigger region
        mask[:, pos_y : pos_y + size_y, pos_x : pos_x + size_x] = 1

        return mask

    def _generate_trigger(self, image_shape: tuple[int, int, int]) -> torch.Tensor:
        """
        Generates the solid trigger pattern.

        Args:
            image_shape (tuple[int, int, int]): Shape of the input image (C, H, W).

        Returns:
            torch.Tensor: Trigger pattern of the same shape as the input image.

        Raises:
            ValueError: If the input image shape is invalid.
        """
        if len(image_shape) != 3:
            raise ValueError(f"Input image must have 3 dimensions (C, H, W), but got {len(image_shape)} dimensions.")

        C, H, W = image_shape
        trigger = torch.zeros((C, H, W))

        # get the position and size of the trigger
        pos_x, pos_y = self.position
        size_x, size_y = self.size

        # apply color to each channel in the trigger region
        trigger[:, pos_y : pos_y + size_y, pos_x : pos_x + size_x] = self.color[:, torch.newaxis, torch.newaxis]

        return trigger


class TriggerTypes(Enum):
    """
    Enum for different trigger types used in backdoor attacks.
    """

    SOLID = InjectSolidTrigger


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generate images from "standard normal distribution"
    GRAYSCALE_IMAGE_SIZE = (1, 28, 28)
    RGB_IMAGE_SIZE = (3, 32, 32)
    grayscale_image = torch.rand(size=GRAYSCALE_IMAGE_SIZE)
    rgb_image = torch.rand(size=RGB_IMAGE_SIZE)

    # create solid trigger transforms
    grayscale_solid_trigger = TriggerTypes.SOLID.value(image_shape=grayscale_image.shape, color=(0.0,), size=(6, 6), position=(20, 20))
    rgb_solid_trigger = TriggerTypes.SOLID.value(image_shape=rgb_image.shape, color=(1.0, 1.0, 1.0), size=(6, 6), position=(24, 24))

    # compose transforms
    grayscale_transform = v2.Compose([grayscale_solid_trigger])
    rgb_transform = v2.Compose([rgb_solid_trigger])

    # apply transforms
    triggered_grayscale_image = grayscale_transform(grayscale_image)
    triggered_rgb_image = rgb_transform(rgb_image)

    # move channel dim to last dim to plot using matplotlib
    grayscale_image = grayscale_image.permute(1, 2, 0)
    rgb_image = rgb_image.permute(1, 2, 0)
    triggered_grayscale_image = triggered_grayscale_image.permute(1, 2, 0)
    triggered_rgb_image = triggered_rgb_image.permute(1, 2, 0)

    # plot original and triggered images
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3), layout="compressed")
    fig.suptitle("Apply Solid Triggers")
    axs[0].imshow(grayscale_image, cmap="gray")
    axs[0].set(title=f"Original: {tuple(grayscale_image.shape)}")
    axs[1].imshow(triggered_grayscale_image, cmap="gray")
    axs[1].set(title=f"Triggered: {tuple(triggered_grayscale_image.shape)}")
    axs[2].imshow(rgb_image, cmap="gray")
    axs[2].set(title=f"Original: {tuple(rgb_image.shape)}")
    axs[3].imshow(triggered_rgb_image, cmap="gray")
    axs[3].set(title=f"Triggered: {tuple(triggered_rgb_image.shape)}")
    plt.show()
