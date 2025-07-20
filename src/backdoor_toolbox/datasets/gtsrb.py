import glob
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import GTSRB
from torchvision.transforms import v2


class GTSRB(GTSRB):
    """
    A modified version of the GTSRB dataset with additional attributes for compatibility with other datasets.
    This class includes `data`, `targets`, `classes`, and `classes_to_idx` attributes.
    Inherits from `torchvision.datasets.GTSRB`.
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
        Args:
            root (str): Root directory of the dataset.
            train (bool): Whether to load the training set. Defaults to True.
            download (bool): Whether to download the dataset if not found. Defaults to False.
            transform (v2.Transform | None, optional): Transformations for images. Defaults to None.
            target_transform (v2.Transform | None, optional): Transformations for labels. Defaults to None.
        """
        super().__init__(root, download=download, transform=transform, target_transform=target_transform)

        # Conditionally load training or testing data based on the `train` flag
        if train:
            self.images, self.labels = self._load_train_data()
        else:
            self.images, self.labels = self._load_test_data()

        # Convert the dataset's image list and labels into tensors for consistency with other datasets
        self.data = torch.stack([torch.tensor(img, dtype=torch.uint8) for img in self.images])  # Image data as tensor
        self.targets = torch.tensor(self.labels, dtype=torch.int64)  # Label data as tensor

        # Store classes and class-to-index mappings
        self.classes = list(sorted(set(self.targets.tolist())))
        self.classes_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_train_data(self):
        """
        Custom method to load the training data from the raw GTSRB dataset.
        You can modify the logic to split your data accordingly.

        Returns:
            Tuple: A tuple containing images and labels for the training set.
        """
        images = []
        labels = []
        train_folder = Path(self.root) / "gtsrb/GTSRB/Training"  # Example path for training images

        # Iterate through the class folders and load images
        for label_folder in train_folder.iterdir():
            if label_folder.is_dir():
                label = int(label_folder.name)
                image_paths = list(label_folder.glob("*.ppm"))  # Assuming .ppm format

                for image_path in image_paths:
                    img = Image.open(image_path)
                    images.append(self._process_image(img))  # Process image (resize, etc.)
                    labels.append(label)

        return images, labels

    def _load_test_data(self):
        """
        Custom method to load the testing data from the raw GTSRB dataset.

        Returns:
            Tuple: A tuple containing images and labels for the test set.
        """
        images = []
        labels = []
        test_folder = Path(self.root) / "gtsrb/GTSRB/Training"  # Example path for test images

        # Iterate through the class folders and load images
        for label_folder in test_folder.iterdir():
            if label_folder.is_dir():
                label = int(label_folder.name)
                image_paths = list(label_folder.glob("*.ppm"))  # Assuming .ppm format

                for image_path in image_paths:
                    img = Image.open(image_path)
                    images.append(self._process_image(img))  # Process image (resize, etc.)
                    labels.append(label)

        return images, labels

    def _process_image(self, img: Image) -> torch.Tensor:
        """
        Process the image before adding it to the dataset. You can add resize, convert to grayscale, etc.

        Args:
            img (PIL.Image): Image to be processed.

        Returns:
            torch.Tensor: Processed image as tensor.
        """
        # Example preprocessing: resize image to a fixed size (32x32 for CIFAR-10 style)
        img = img.resize((32, 32))
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.uint8)
        img_tensor = img_tensor.view(3, 32, 32)  # Assuming the image is RGB
        return img_tensor

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Retrieves an image and its label by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the image tensor and its label.
        """
        image, label = self.data[index], self.targets[index]
        return image, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # define parameters
    root = "./data"
    train = True
    download = True

    # initialize dataset
    dataset = GTSRB(root=root, train=train, download=download)

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
