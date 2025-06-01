import random
from enum import Enum

import torch
import torch.nn.functional as F

# from PIL import Image
from torchvision.transforms import v2, functional as tf
from torchvision.io import read_image

from backdoor_toolbox.triggers.transform.base import InjectTrigger


class InjectSolidTrigger(InjectTrigger):
    """
    Class for injecting a solid-colored trigger into an image.

    Attributes:
        color (torch.Tensor): Color of the trigger (tuple of floats, one for each channel).
        size (tuple[int, int]): Size of the trigger (width, height).
        position (tuple[int, int]): Position of the top-left corner of the trigger (x, y).
    """

    name = "SOLID"

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
            raise ValueError(
                f"Number of channels of `image_shape`:{image_shape[0]} is not equal to length of `self.color`:{len(self.color)}"
            )

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


class InjectPatternTrigger(InjectTrigger):
    """
    Class for injecting a patterned (checkerboard) trigger into an image.

    Attributes:
        color (tuple[float, float]): Two alternating colors for the pattern.
        size (tuple[int, int]): Size of the trigger (width, height).
        position (tuple[int, int]): Position of the top-left corner of the trigger (x, y).
    """

    name = "PATTERN"

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        color: tuple[float, float] = (0.0, 1.0),
        size: tuple[int, int] = (6, 6),
        position: tuple[int, int] = (0, 0),
    ):
        super().__init__(image_shape)
        self.color = color
        self.size = size
        self.position = position

        if len(self.color) != 2:
            raise ValueError("colors must be a tuple of 2 floats (e.g., (0.0, 1.0)).")
        if len(self.size) != 2:
            raise ValueError("size must be a tuple of 2 elements (width, height).")
        if len(self.position) != 2:
            raise ValueError("position must be a tuple of 2 elements (x, y).")

    def _generate_mask(self, image_shape: tuple[int, int, int]) -> torch.Tensor:
        C, H, W = self.image_shape
        mask = torch.zeros((C, H, W))
        pos_x, pos_y = self.position
        size_x, size_y = self.size
        mask[:, pos_y : pos_y + size_y, pos_x : pos_x + size_x] = 1
        return mask

    def _generate_trigger(self, image_shape: tuple[int, int, int]) -> torch.Tensor:
        C, H, W = self.image_shape
        trigger = torch.zeros((C, H, W))
        pos_x, pos_y = self.position
        size_x, size_y = self.size
        # Create a checkerboard pattern in the region.
        pattern = torch.zeros((size_y, size_x))
        for i in range(size_y):
            for j in range(size_x):
                pattern[i, j] = self.color[(i + j) % 2]
        # Replicate the pattern to all channels if needed.
        if C > 1:
            pattern = pattern.unsqueeze(0).repeat(C, 1, 1)
        else:
            pattern = pattern.unsqueeze(0)
        trigger[:, pos_y : pos_y + size_y, pos_x : pos_x + size_x] = pattern
        return trigger


class InjectNoiseTrigger(InjectTrigger):
    """
    Class for injecting a noise-based trigger into an image.

    Attributes:
        noise_level (float): Scaling factor for the noise intensity.
        size (tuple[int, int]): Size of the trigger (width, height).
        position (tuple[int, int]): Position of the top-left corner of the trigger (x, y).
        seed (int): Seed for reproducibility.
    """

    name = "Noise"

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        noise_range: float = 1.0,
        size: tuple[int, int] = (6, 6),
        position: tuple[int, int] = (0, 0),
        seed: int = 42,
    ):
        super().__init__(image_shape)
        self.noise_range = noise_range
        self.size = size
        self.position = position
        self.seed = seed

        if len(self.size) != 2:
            raise ValueError("size must be a tuple of 2 elements (width, height).")
        if len(self.position) != 2:
            raise ValueError("position must be a tuple of 2 elements (x, y).")

    def _generate_mask(self, image_shape: tuple[int, int, int]) -> torch.Tensor:
        C, H, W = self.image_shape
        mask = torch.zeros((C, H, W))
        pos_x, pos_y = self.position
        size_x, size_y = self.size
        mask[:, pos_y : pos_y + size_y, pos_x : pos_x + size_x] = 1
        return mask

    def _generate_trigger(self, image_shape: tuple[int, int, int]) -> torch.Tensor:
        C, H, W = self.image_shape
        trigger = torch.zeros((C, H, W))
        pos_x, pos_y = self.position
        size_x, size_y = self.size
        # Set random seed for reproducibility.
        torch.manual_seed(self.seed)
        # Generate noise within the trigger region.
        noise = torch.empty((C, size_y, size_x)).uniform_(*self.noise_range)
        trigger[:, pos_y : pos_y + size_y, pos_x : pos_x + size_x] = noise
        return trigger


class InjectBlendTrigger(InjectTrigger):
    """
    Class for injecting a full-image transparent trigger into an image.
    The trigger is blended with the original image using an alpha factor.

    Attributes:
        trigger_image (torch.Tensor): The full-sized trigger image to be blended.
        size (tuple[int, int]): Placeholder for compatibility.
        position (tuple[int, int]): Placeholder for compatibility.
        alpha (float): Transparency factor for blending (0: fully transparent, 1: fully visible).
    """

    name = "Blend"

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        trigger_image: torch.Tensor,
        alpha: float = 0.5,
    ):
        super().__init__(image_shape)
        self.alpha = alpha
        self.image_shape = image_shape  # (C, H, W)
        self.trigger_image = self._resize_trigger(trigger_image)

        # Store size and position for compatibility (but they are unused)
        self.size = (image_shape[1], image_shape[2])  # Full image size
        self.position = (0, 0)  # Always starts from (0,0)

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in the range [0, 1].")

    def _resize_trigger(self, trigger_image: torch.Tensor) -> torch.Tensor:
        """
        Resizes the trigger image to match the input image shape if needed.
        """
        C, H, W = self.image_shape

        if trigger_image.shape != (C, H, W):
            trigger_image = F.interpolate(trigger_image.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0)

        return trigger_image

    def _generate_mask(self) -> torch.Tensor:
        """
        Generates a full-image mask filled with 1s (since the entire image is modified).
        """
        C, H, W = self.image_shape
        return torch.ones((C, H, W), dtype=torch.float32)

    def _generate_trigger(self) -> torch.Tensor:
        """
        Returns the full-image trigger (already resized if needed).
        """
        return self.trigger_image

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the transparent trigger by blending the full-image trigger with the original image.
        """
        mask = self._generate_mask()
        trigger = self._generate_trigger()
        return (1 - self.alpha * mask) * image + (self.alpha * mask) * trigger


class TriggerTypes(Enum):
    """
    Enum for different trigger types used in backdoor attacks.
    """

    SOLID = InjectSolidTrigger
    PATTERN = InjectPatternTrigger
    NOISE = InjectNoiseTrigger
    BLEND = InjectBlendTrigger


class TriggerSelector:
    """
    Policy for generating N random backdoor triggers.

    Attributes:
        image_shape (tuple[int, int, int]): Image shape (C, H, W).
        trigger_types (list): List of available trigger classes.
        num_attacks (int): Number of distinct triggers to generate.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        trigger_types: list,
        num_triggers: int,
        blend_images: list,
        seed: int = 42,
    ):
        self.image_shape = image_shape
        self.trigger_types = trigger_types
        self.num_triggers = num_triggers
        self.blend_images = blend_images
        random.seed(seed)
        torch.manual_seed(seed)

    def _random_color(self, num_channels: int) -> tuple[float, ...]:
        """Generates a random color tuple with values in [0, 1]."""
        return tuple(random.random() for _ in range(num_channels))

    def _random_size(self) -> tuple[int, int]:
        """
        Generates a random size for the trigger based on a fraction of the image dimensions.
        For example, size will be between 5% and 20% of image width and height.
        """
        _, H, W = self.image_shape
        size = random.randint(3, max(1, int(0.2 * W)))
        return (size, size)

    def _random_position(self, size: tuple[int, int]) -> tuple[int, int]:
        """
        Generates a random position (top-left corner) for the trigger near the image borders,
        ensuring at least a 1-pixel offset from the edges.
        """
        _, H, W = self.image_shape
        size_x, size_y = size

        border_margin = max(2, int(0.2 * min(H, W)))  # Define a border width (at least 2 pixels)

        # Select from the border regions but with a 1-pixel offset from the absolute edge
        pos_x = random.choice(
            [
                random.randint(1, border_margin - 1),  # Left border (not at 0)
                random.randint(W - size_x - border_margin + 1, W - size_x - 1),  # Right border (not at W - size_x)
            ]
        )

        pos_y = random.choice(
            [
                random.randint(1, border_margin - 1),  # Top border (not at 0)
                random.randint(H - size_y - border_margin + 1, H - size_y - 1),  # Bottom border (not at H - size_y)
            ]
        )

        return (pos_x, pos_y)

    def get_triggers(self) -> list:
        """
        Generates and returns a list of trigger instances with randomized parameters.
        """
        triggers = []
        C, _, _ = self.image_shape

        for _ in range(self.num_triggers):
            # Randomly select a trigger type.
            trigger_cls = random.choice(self.trigger_types)

            # Generate random size and position.
            size = self._random_size()
            position = self._random_position(size)

            # Depending on trigger type, randomize parameters.
            if trigger_cls.__name__ == "InjectSolidTrigger":
                color = self._random_color(C)
                trigger = trigger_cls(self.image_shape, color=color, size=size, position=position)

            elif trigger_cls.__name__ == "InjectPatternTrigger":
                # Pattern trigger requires two alternating colors.
                colors = (random.random(), random.random())
                trigger = trigger_cls(self.image_shape, color=colors, size=size, position=position)

            elif trigger_cls.__name__ == "InjectNoiseTrigger":
                trigger = trigger_cls(self.image_shape, noise_range=(0.2, 1.0), size=size, position=position)

            elif trigger_cls.__name__ == "InjectBlendTrigger":
                color = self._random_color(C)
                alpha = random.uniform(0.05, 0.15)
                trigger_image = random.choice(self.blend_images)
                self.blend_images.remove(trigger_image)
                trigger = trigger_cls(self.image_shape, trigger_image=trigger_image, alpha=alpha)

            else:
                # Fallback: use a solid trigger.
                color = self._random_color(C)
                trigger = InjectSolidTrigger(self.image_shape, color=color, size=size, position=position)

            triggers.append(trigger)

        return triggers


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ROW = 3
    COL = 4
    NUM_ATTACKS = ROW * COL
    SEED = 0
    TRIGGERS = (InjectSolidTrigger, InjectPatternTrigger, InjectNoiseTrigger, InjectBlendTrigger)

    # import a trigger image
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    trigger_image_1 = transform(tf.rgb_to_grayscale(read_image(r"./assets/blend_trigger/noise.jpg")))
    trigger_image_2 = transform(tf.rgb_to_grayscale(read_image(r"./assets/blend_trigger/kitty.jpg")))

    # solid image
    GRAYSCALE_IMAGE_SIZE = (1, 28, 28)
    transform = v2.Compose(
        [
            v2.Resize((GRAYSCALE_IMAGE_SIZE[1], GRAYSCALE_IMAGE_SIZE[2])),
            # v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    grayscale_image = torch.full(size=GRAYSCALE_IMAGE_SIZE, fill_value=0.5, dtype=torch.float32)
    grayscale_policies = TriggerSelector(
        image_shape=GRAYSCALE_IMAGE_SIZE,
        trigger_types=TRIGGERS,
        num_triggers=NUM_ATTACKS,
        blend_images=[trigger_image_1, trigger_image_2],
        seed=SEED,
    ).get_triggers()
    grayscale_images_transformed = [policy(grayscale_image).permute(1, 2, 0) for policy in grayscale_policies]

    # plot original and triggered images
    fig, axs = plt.subplots(nrows=ROW, ncols=COL, figsize=(4 * COL, 4 * ROW))
    for r in range(ROW):
        for c in range(COL):
            axs[r, c].imshow(grayscale_images_transformed[COL * r + c], cmap="gray", vmin=0, vmax=1)
            axs[r, c].set_title(
                f"{grayscale_policies[COL * r + c].__class__.__name__}-{grayscale_policies[COL * r + c].size}-{grayscale_policies[COL * r + c].position}"
            )
            axs[r, c].axis("off")
    plt.show()

    # import a trigger image
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    trigger_image_1 = transform(read_image(r"./assets/blend_trigger/noise.jpg"))
    trigger_image_2 = transform(read_image(r"./assets/blend_trigger/kitty.jpg"))

    # solid image
    RGB_IMAGE_SIZE = (3, 28, 28)
    transform = v2.Compose(
        [
            v2.Resize((RGB_IMAGE_SIZE[1], RGB_IMAGE_SIZE[2])),
            # v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    rgb_image = torch.full(size=RGB_IMAGE_SIZE, fill_value=0.5, dtype=torch.float32)
    rgb_policies = TriggerSelector(
        image_shape=RGB_IMAGE_SIZE,
        trigger_types=TRIGGERS,
        num_triggers=NUM_ATTACKS,
        blend_images=[trigger_image_1, trigger_image_2],
        seed=SEED,
    ).get_triggers()
    rgb_images_transformed = [policy(rgb_image).permute(1, 2, 0) for policy in rgb_policies]

    # plot original and triggered images
    fig, axs = plt.subplots(nrows=ROW, ncols=COL, figsize=(4 * COL, 4 * ROW))
    for r in range(ROW):
        for c in range(COL):
            axs[r, c].imshow(rgb_images_transformed[COL * r + c], cmap="gray", vmin=0, vmax=1)
            axs[r, c].set_title(
                f"{rgb_policies[COL * r + c].__class__.__name__}-{rgb_policies[COL * r + c].size}-{rgb_policies[COL * r + c].position}"
            )
            axs[r, c].axis("off")
    plt.show()
