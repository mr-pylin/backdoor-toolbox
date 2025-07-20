import math
import random
from enum import Enum

import torch
import torch.nn.functional as F
from torchvision.io import read_image

# from PIL import Image
from torchvision.transforms import functional as tf
from torchvision.transforms import v2

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
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        trigger_types: list[type],
        num_triggers: int,
        blend_images: list[torch.Tensor],
        *,
        seed,
        num_similarity: int = 0,
        similarity_ratio: float = 0.0,
    ):
        self.image_shape = image_shape
        self.trigger_types = trigger_types
        self.num_triggers = num_triggers
        self.blend_images = blend_images.copy()
        self._all_blend_images = blend_images.copy()
        self.num_similarity = num_similarity
        self.similarity_ratio = similarity_ratio
        random.seed(seed)
        torch.manual_seed(seed)

    def _random_color(self, num_channels: int) -> tuple[float, ...]:
        return tuple(random.random() for _ in range(num_channels))

    def _random_size(self) -> tuple[int, int]:
        _, H, W = self.image_shape
        size_x = random.randint(3, max(1, int(0.2 * W)))
        size_y = random.randint(3, max(1, int(0.2 * H)))
        return (size_x, size_y)

    def _random_position(self, size: tuple[int, int]) -> tuple[int, int]:
        _, H, W = self.image_shape
        size_x, size_y = size
        border = max(2, int(0.2 * min(H, W)))
        pos_x = random.choice(
            [
                random.randint(1, border - 1),
                random.randint(W - size_x - border + 1, W - size_x - 1),
            ]
        )
        pos_y = random.choice(
            [
                random.randint(1, border - 1),
                random.randint(H - size_y - border + 1, H - size_y - 1),
            ]
        )
        return (pos_x, pos_y)

    def _make_random(self, cls: type) -> torch.nn.Module:
        C, _, _ = self.image_shape
        if cls.__name__ == "InjectBlendTrigger":
            if not self.blend_images:
                self.blend_images = self._all_blend_images.copy()
            alpha = random.uniform(0.05, 0.1)
            # alpha = random.uniform(0.15, 0.2)  # for cifar10
            idx = random.randrange(len(self.blend_images))
            trigger_img = self.blend_images.pop(idx)
            return cls(self.image_shape, trigger_image=trigger_img, alpha=alpha)

        size = self._random_size()
        pos = self._random_position(size)

        if cls.__name__ == "InjectSolidTrigger":
            color = self._random_color(C)
            return cls(self.image_shape, color=color, size=size, position=pos)
        if cls.__name__ == "InjectPatternTrigger":
            levels = 64  # 1/64 = 0.015625 steps
            a = random.randint(0, levels - 1)
            b = random.randint(0, levels - 2)
            if b >= a:
                b += 1  # ensure b ≠ a
            colors = (a / (levels - 1), b / (levels - 1))
            return cls(self.image_shape, color=colors, size=size, position=pos)
        if cls.__name__ == "InjectNoiseTrigger":
            return cls(self.image_shape, noise_range=(0.2, 1.0), size=size, position=pos)
        # fallback
        color = self._random_color(C)
        return InjectSolidTrigger(self.image_shape, color=color, size=size, position=pos)

    def _perturb(self, trig: torch.nn.Module, ratio: float) -> torch.nn.Module:
        """
        ratio=1.0 => identical; ratio=0.0 => fully random within allowed bounds
        """
        cls = trig.__class__
        C, H, W = self.image_shape

        # size
        sx, sy = trig.size
        rand_sx, rand_sy = self._random_size()
        new_sx = int(sx * ratio + rand_sx * (1 - ratio))
        new_sy = int(sy * ratio + rand_sy * (1 - ratio))
        new_size = (min(max(1, new_sx), W), min(max(1, new_sy), H))

        # position drift bounded by size (like a kernel around original pos)
        px, py = trig.position
        max_dx = new_size[0] + sx
        max_dy = new_size[1] + sy

        drift_x = (1 - ratio) * random.randint(-max_dx, max_dx)
        drift_x = int(math.ceil(drift_x) if drift_x > 0 else math.floor(drift_x))
        drift_y = (1 - ratio) * random.randint(-max_dy, max_dy)
        drift_y = int(math.ceil(drift_y) if drift_y > 0 else math.floor(drift_y))

        new_px = min(max(0, px + drift_x), W - new_size[0])
        new_py = min(max(0, py + drift_y), H - new_size[1])
        new_pos = (new_px, new_py)

        kwargs = {"image_shape": self.image_shape, "size": new_size, "position": new_pos}
        if cls.__name__ == "InjectSolidTrigger":
            oc = trig.color.tolist()
            rc = [random.random() for _ in oc]
            new_c = tuple(o * ratio + r * (1 - ratio) for o, r in zip(oc, rc))
            kwargs["color"] = new_c
        elif cls.__name__ == "InjectPatternTrigger":
            c0, c1 = trig.color
            # rc0, rc1 = random.random(), random.random()

            levels = 64  # 1/64 = 0.015625 steps
            a = random.randint(0, levels - 1)
            b = random.randint(0, levels - 2)
            if b >= a:
                b += 1  # ensure b ≠ a
            rc0, rc1 = (a / (levels - 1), b / (levels - 1))

            kwargs["color"] = (c0 * ratio + rc0 * (1 - ratio), c1 * ratio + rc1 * (1 - ratio))

        return cls(**kwargs)

    def get_triggers(self) -> list[torch.nn.Module]:
        triggers: list[torch.nn.Module] = []
        sim_base = None

        # similar group (exclude blend)
        if self.num_similarity > 0:
            non_blend = [
                t for t in self.trigger_types if t.__name__ not in {"InjectBlendTrigger", "InjectNoiseTrigger"}
            ]

            if not non_blend:
                raise ValueError("No non-blend types for similarity")
            sim_base = random.choice(non_blend)
            base = self._make_random(sim_base)
            triggers.append(base)
            for _ in range(self.num_similarity - 1):
                triggers.append(self._perturb(base, self.similarity_ratio))

        # remaining: uniform over others
        rem = self.num_triggers - len(triggers)
        if rem > 0:
            if sim_base:
                cands = [t for t in self.trigger_types if t is not sim_base]
            else:
                cands = list(self.trigger_types)
            n = len(cands)
            q, r = divmod(rem, n)
            for i, cls in enumerate(cands):
                cnt = q + (1 if i < r else 0)
                for _ in range(cnt):
                    triggers.append(self._make_random(cls))

        random.shuffle(triggers)
        return triggers


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image

    # load an example image and prepare blend images list
    img1 = Image.open("assets/blend_trigger/kitty.jpg").convert("RGB")  # replace with your image path
    img2 = Image.open("assets/blend_trigger/creeper.jpg").convert("RGB")
    to_tensor = T.ToTensor()
    image_tensor1 = to_tensor(img1)
    image_tensor2 = to_tensor(img2)  # shape: C, H, W

    # suppose we have some images for blending
    blend_imgs = [image_tensor1, image_tensor2]  # dummy blend images

    selector = TriggerSelector(
        image_shape=(3, 32, 32),
        trigger_types=[InjectSolidTrigger, InjectPatternTrigger, InjectNoiseTrigger, InjectBlendTrigger],
        num_triggers=7,
        blend_images=blend_imgs,
        seed=1,
        num_similarity=6,
        similarity_ratio=0.75,
    )

    triggers = selector.get_triggers()
    # apply triggers to a copy of the image
    applied = []
    for trig in triggers:
        img_copy = torch.ones((3, 32, 32))
        masked = trig(img_copy)
        applied.append(masked)

    # for visualization, convert back to PIL and save
    to_pil = T.ToPILImage()
    for idx, tens in enumerate(applied):
        pil_img = to_pil(tens)
        pil_img.save(f"output_trigger_{idx}.png")
