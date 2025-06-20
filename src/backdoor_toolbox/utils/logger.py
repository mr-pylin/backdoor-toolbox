import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import write_png
from torchvision.transforms import v2


class Logger:
    """
    Utility for saving hyperparameters, metrics, images, feature maps, model weights, etc.

    Attributes:
      root: Base directory for all outputs.
      verbose: Whether to print status messages.
    """

    def __init__(self, root: Path, sub_root: str | None, verbose: bool = True):
        """
        Initialize the Logger instance.

        Creates a timestamped directory under the specified root path for logging outputs.

        Args:
            root (Path): Root directory where the log folder will be created.
            subroot (str): Sub-Root directory where the log folder will be created
            verbose (bool, optional): If True, enables console output during logging operations. Defaults to True.
        """
        if sub_root is None:
            self.root = Path(f"{root}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}")
        else:
            self.root = Path(f"{root}/{sub_root}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}")
        self.verbose = verbose

        # Make directory if not available
        self.root.mkdir(parents=True, exist_ok=True)

    def save_configs(self, src_path: Path, dst_path: Path, filename: str) -> None:
        """
        Save a Python configuration file into the logging directory.

        Copies the file `<filename>.py` from `src_path` and places it under
        `self.root/dst_path`. The directory structure is created if it does not exist.

        Args:
            src_path (Path): Path to the source directory containing the `.py` file.
            dst_path (Path): Relative subdirectory under the logger root where the file will be saved.
            filename (str): Name of the configuration file (without `.py` extension).
        """
        save_dir = self.root / dst_path
        save_dir.mkdir(parents=True, exist_ok=True)

        src_file = src_path / f"{filename}.py"
        dst_file = save_dir / f"{filename}.py"

        try:
            shutil.copyfile(src_file, dst_file)
            if self.verbose:
                print(f"[Logger] Saved a copy of '{src_file}' to '{dst_file}'")

        except FileNotFoundError:
            print(f"[Logger][Error] Source file not found: '{src_file}'")

        except PermissionError:
            print(f"[Logger][Error] Permission denied when accessing '{src_file}' or '{dst_file}'")

        except Exception as e:
            print(f"[Logger][Error] Unexpected error while saving config: {e}")

    def save_hyperparameters(self, path: Path, filename: str, **hyperparameters: Any) -> None:
        """
        Save hyperparameters to disk.

        Stores JSON-serializable hyperparameters in a `.json` file and saves non-serializable
        items (e.g., PyTorch `state_dict`s) separately using `torch.save`.

        Args:
            path (Path): Subdirectory under the logger root where files will be saved.
            filename (str): Base filename (without extension) for the saved files.
            **hyperparameters (Any): Arbitrary keyword arguments representing hyperparameters.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{filename}.json"
        state_dict_path = save_dir / f"{filename}_state_dict.pth"

        # Separate serializable and non-serializable hyperparameters
        serializable_hparams = {}
        non_serializable_hparams = {}

        for key, value in hyperparameters.items():
            try:
                # Try to serialize the value to JSON
                json.dumps(value)
                serializable_hparams[key] = value
            except (TypeError, ValueError):
                # If serialization fails, treat as non-serializable and save separately
                non_serializable_hparams[key] = value

        try:
            # Save serializable hyperparameters to JSON
            with save_path.open("w") as f:
                json.dump(serializable_hparams, f, indent=4)

            # Save non-serializable hyperparameters (e.g., state_dict) using torch.save
            if non_serializable_hparams:
                torch.save(non_serializable_hparams, state_dict_path)

            if self.verbose:
                print(f"[Logger] Hyperparameters saved to '{save_path}'")
                if non_serializable_hparams:
                    print(f"[Logger] Non-serializable hyperparameters saved to '{state_dict_path}'")

        except Exception as e:
            print(f"[Logger][Error] Unexpected error while saving hyperparameters: {e}")

    def save_metrics(self, path: Path, filename: str, **data: float) -> None:
        """
        Save metric data to a CSV file.

        Appends metric key-value pairs to a CSV file located in the specified subdirectory.
        Creates the directory and file if they do not exist.

        Args:
            path (Path): Subdirectory under the logger root for saving the file.
            filename (str): CSV filename without the `.csv` extension.
            **data (float): Arbitrary metric key-value pairs to log.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.csv"

        file_exists = save_path.exists()

        try:
            with save_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(data)

            if self.verbose:
                print(f"[Logger] Metrics appended to '{save_path}'")

        except Exception as e:
            print(f"[Logger][Error] Failed to save metrics: {e}")

    def save_labeled_matrix(
        self,
        path: Path,
        filename: str,
        matrix: np.ndarray | torch.Tensor,
        row0_col0_title: str,
        row_labels: list[int | str],
        col_labels: list[int | str] | None = None,
    ) -> None:
        """
        Save a labeled 2D matrix as a CSV file.

        This function saves matrices such as confusion matrices or cross-metrics
        with labeled rows and columns, including a custom header title for the
        top-left cell.

        Args:
            path (Path): Subdirectory under the logger root to save the file.
            filename (str): Filename without the `.csv` extension.
            matrix (np.ndarray | torch.Tensor): 2D matrix data to save.
            row0_col0_title (str): Title for the top-left header cell (e.g., "True/Pred").
            row_labels (Sequence[int | str]): Labels for the rows.
            col_labels (Optional[Sequence[int | str]], optional): Labels for the columns.
                Defaults to `row_labels` if not provided.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.csv"

        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()

        if col_labels is None:
            col_labels = row_labels

        # Start the header with row0_col0_title and col_labels
        header = [row0_col0_title] + list(col_labels)

        # Write the data to a CSV file
        with open(save_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(header)

            # Write rows with row_labels and matrix data
            for i, row in enumerate(matrix):
                writer.writerow([row_labels[i]] + list(row))

        if self.verbose:
            print(f"[Logger] Matrix saved to '{save_path}'")

    def plot_and_save_metrics(
        self,
        path: Path,
        filename: str,
        save_format: str,
        ylabel: str,
        title: str,
        data: dict[str, list[float]],
        xlabel: str = "Epochs",
        show: bool = False,
        ylim: tuple[float, float] | None = None,
        markers: bool = False,
    ) -> None:
        """
        Plot and save line graphs for one or more metric series over epochs.

        Args:
            path (Path): Directory under the logger root to save the plot.
            filename (str): Name of the file (without extension).
            save_format (str): Image format to save ('png' or 'svg').
            ylabel (str): Label for the Y-axis.
            title (str): Title of the plot.
            data (dict[str, list[float]]): Dictionary mapping series labels to Y-axis data.
            xlabel (str, optional): Label for the X-axis. Defaults to "Epochs".
            show (bool, optional): Whether to display the plot interactively. Defaults to False.
            ylim (Optional[tuple[float, float]], optional): Y-axis limits as (min, max). Defaults to None.
            markers (bool, optional): Whether to add markers on data points. Defaults to False.

        Raises:
            ValueError: If `save_format` is not 'png' or 'svg'.
        """
        if save_format not in {"png", "svg"}:
            raise ValueError("Invalid save_format. Only 'png' and 'svg' are supported.")

        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.{save_format}"

        plt.figure(figsize=(12, 9), layout="compressed")

        max_len = max(len(values) for values in data.values())
        for label, values in data.items():
            plt.plot(range(1, len(values) + 1), values, label=label, marker="o" if markers else None)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(range(1, max_len + 1))
        plt.legend(loc="best")

        if ylim:
            plt.ylim(*ylim)

        plt.savefig(save_path, format=save_format, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

        if self.verbose:
            print(f"[Logger] Plot saved to '{save_path}'")

    def save_weights(
        self,
        path: Path,
        filename: str,
        model: nn.Module,
        epoch: int,
        only_state_dict: bool = True,
    ) -> None:
        """
        Save model weights to a `.pth` file, including the epoch number in the filename.

        Args:
            path (Path): Subdirectory under the logger root to save the weights.
            filename (str): Base filename without extension.
            model (nn.Module): PyTorch model instance.
            epoch (int): Epoch number to include in the filename.
            only_state_dict (bool, optional): If True, save only the model's state dictionary.
                If False, save the entire model. Defaults to True.

        Raises:
            Exception: Propagates exceptions encountered during saving.
        """

        original_device = next(model.parameters()).device

        # Update filename to include epoch number
        filename_with_epoch = f"{filename}_epoch_{epoch}"

        # Ensure the directory exists
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)

        # Define full save path with '.pth' extension
        save_path = save_dir / f"{filename_with_epoch}.pth"

        try:
            # Save either the entire model or just the state dict
            if only_state_dict:
                torch.save(model.cpu().state_dict(), save_path)
            else:
                torch.save(model.cpu(), save_path)

            if self.verbose:
                print(f"[Logger] Model weights saved to '{save_path}'")

            model.to(original_device)  # restore model to original device

        except Exception as e:
            print(f"[Error]: Failed to save model weights. Error: {e}")
            raise

    def save_image_predictions(
        self,
        path: Path,
        filename: str,
        model: nn.Module,
        dataset: Dataset,
        nrows: int,
        ncols: int,
        save_grid: bool,
        show_grid: bool,
        clamp: bool = True,
    ) -> None:
        """
        Save individual images with predicted and true labels as filenames.
        Optionally, save and display a grid of the first (nrows × ncols) predictions.

        Args:
            path (Path): Directory to save images.
            filename (str): Base filename for the grid image.
            model (nn.Module): Trained PyTorch model.
            dataset (Dataset): Dataset to sample images from.
            nrows (int): Number of rows in the grid.
            ncols (int): Number of columns in the grid.
            save_grid (bool): Whether to save a grid image of predictions.
            show_grid (bool): Whether to display the grid image.
            clamp (bool, optional): Clamp float images to [0,1] before saving. Defaults to True.
        """
        original_device = next(model.parameters()).device

        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)

        model.cpu().eval()

        batch_size = nrows * ncols if save_grid else 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        data = next(iter(dataloader))
        images, targets = data[0], data[1]
        preds = model(images.cpu()).argmax(dim=1).cpu()
        model.to(original_device)  # restore model to original device

        # Save individual images
        for i, (img, label, pred) in enumerate(zip(images, targets, preds)):
            if clamp and torch.is_floating_point(img):
                img = img.clamp(0, 1)
            img_to_save = (img * 255).to(torch.uint8)
            save_path = save_dir / filename / f"img_{i:04d}_true_{label}_pred_{pred}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            write_png(img_to_save, save_path)

        if save_grid:
            _, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5), layout="constrained")
            plt.suptitle(f"First {nrows * ncols} predictions")

            for i in range(nrows):
                for j in range(ncols):
                    idx = i * ncols + j
                    img = images[idx]
                    if clamp and torch.is_floating_point(img):
                        img = img.clamp(0, 1)
                    axs[i, j].imshow(img.permute(1, 2, 0), cmap="gray")
                    axs[i, j].set_title(f"t:{targets[idx]}, p:{preds[idx]}")
                    axs[i, j].axis("off")

            grid_path = save_dir / f"{filename}_overview.png"
            plt.savefig(grid_path, format="png", bbox_inches="tight")

            if show_grid:
                plt.show()

            plt.close()

        if self.verbose:
            print(f"[Logger] Saved prediction images to '{save_dir}'")
            if save_grid:
                print(f"[Logger] Saved grid image to '{grid_path}'")

    def save_trigger_pattern(
        self,
        path: Path,
        filename: str,
        trigger_policy: v2.Transform,
        bg_size: tuple[int, int, int],
        bg_color: float,
        dataset: Dataset | None = None,
        n_samples: int = 0,
        clamp: bool = True,
        show: bool = False,
    ) -> None:
        """
        Save a trigger pattern on a uniform background and optionally on real dataset samples.

        Args:
            path (Path): Directory to save images.
            filename (str): Base filename (without extension).
            trigger_policy (v2.Transform): Trigger transform with `.name` attribute.
            bg_size (tuple[int, int, int]): Background shape (C, H, W).
            bg_color (float): Background fill color in [0, 1].
            dataset (Optional[Dataset]): Optional dataset to sample real images.
            n_samples (int): Number of real images to apply the trigger to.
            clamp (bool): Clamp float images to [0, 1] before saving.
            show (bool): Whether to display the background image.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the uniform background trigger visualization
        bg = torch.full(size=bg_size, fill_value=bg_color, dtype=torch.float32)
        poisoned_bg = trigger_policy(bg)
        if clamp and torch.is_floating_point(poisoned_bg):
            poisoned_bg = poisoned_bg.clamp(0, 1)

        demo_path = save_dir / f"{filename}_{trigger_policy.name}_{bg_size[1]}x{bg_size[2]}.png"
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        img_to_save = (poisoned_bg * 255).to(torch.uint8)
        write_png(img_to_save, demo_path)

        if show:
            plt.imshow(poisoned_bg.permute(1, 2, 0), cmap="gray")
            plt.title(f"Trigger: {trigger_policy.name}")
            plt.axis("off")
            plt.show()

        if self.verbose:
            print(f"[Logger] Trigger background image saved to '{demo_path}'")

        # Optionally save triggered versions of real dataset samples
        if dataset and n_samples > 0:
            subdir = save_dir / filename
            subdir.mkdir(parents=True, exist_ok=True)

            dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
            images, labels = next(iter(dataloader))

            for i in range(min(n_samples, len(images))):
                img = images[i]
                label = labels[i]

                poisoned_img = trigger_policy(img)
                if clamp and torch.is_floating_point(poisoned_img):
                    poisoned_img = poisoned_img.clamp(0, 1)

                img_to_save = (poisoned_img * 255).to(torch.uint8)
                save_path = subdir / f"img_{i:04d}_label_{label}_triggered.png"
                write_png(img_to_save, save_path)

            if self.verbose:
                print(f"[Logger] Saved {min(n_samples, len(images))} triggered samples to '{subdir}'")

    def save_feature_maps(
        self,
        path: Path,
        feature_dict: dict[str, torch.Tensor],
        normalize: bool = True,
        aggregation: bool = False,
        overview: bool = False,
    ) -> None:
        """
        Save feature maps from multiple layers to disk as PNG images.

        Args:
            path (Path): Directory where feature maps will be saved.
            feature_dict (dict[str, torch.Tensor]): Mapping from layer names to feature tensors.
                Each tensor shape: (batch_size, channels, height, width).
            normalize (bool): Whether to normalize each feature map to [0, 1] before saving.
            aggregation (bool): Whether to save aggregated (mean) feature maps per sample.
            overview (bool): Whether to save overview grid plots of all feature maps per sample.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)

        total_saved = 0

        for layer_name, feature_tensor in feature_dict.items():
            layer_path = save_dir / layer_name
            layer_path.mkdir(parents=True, exist_ok=True)

            N, A, H, W = feature_tensor.shape  # batch, channels, height, width
            fig_cols = math.ceil(math.sqrt(A))
            fig_rows = math.ceil(A / fig_cols)

            for n in range(N):
                sample_path = layer_path / f"sample_{n}"
                sample_path.mkdir(parents=True, exist_ok=True)

                for a in range(A):
                    fmap = feature_tensor[n, a]  # shape (H, W)
                    if normalize:
                        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
                    fmap_byte = (fmap * 255).byte().unsqueeze(0)  # shape (1, H, W)
                    write_png(fmap_byte, sample_path / f"feature_{a}.png")
                    total_saved += 1

                # Save aggregation plot per sample
                if aggregation:
                    fmap = feature_tensor[n].mean(dim=0)
                    if normalize:
                        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
                    fmap_byte = (fmap * 255).byte().unsqueeze(0)  # shape (1, H, W)
                    write_png(fmap_byte, sample_path / f"mean_{layer_name}_sample_{n}.png")

                # Save overview plot per sample
                if overview:
                    fig, axes = plt.subplots(
                        fig_rows,
                        fig_cols,
                        figsize=(fig_cols * 2, fig_rows * 2),
                        layout="compressed",
                    )
                    axes = axes.flatten()

                    for a in range(A):
                        fmap = feature_tensor[n, a]
                        if normalize:
                            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
                        axes[a].imshow(fmap.cpu(), cmap="gray")
                        axes[a].axis("off")
                        axes[a].set_title(f"Map {a}")

                    for a in range(A, len(axes)):
                        axes[a].axis("off")

                    fig.suptitle(f"{layer_name} | Sample {n}")
                    plt.tight_layout()
                    overview_path = sample_path / f"overview_{layer_name}_sample_{n}.png"
                    plt.savefig(overview_path, dpi=96)
                    plt.close(fig)

                    if self.verbose:
                        print(f"[Logger] Overview image for '{layer_name}', sample {n} saved to '{overview_path}'")

        if self.verbose:
            print(f"[Logger] Saved {total_saved} feature map images to '{save_dir}'")

    def save_heatmaps(
        self,
        path: Path,
        overlays_dict: dict[str, torch.Tensor],
        normalize: bool = True,
        overview: bool = False,
    ) -> None:
        """
        Save heatmaps (e.g., Grad-CAM overlays) as PNG images per sample, optionally with an overview grid.

        Args:
            path (Path): Directory to save heatmaps.
            overlays_dict (dict[str, torch.Tensor]): Mapping from layer names to heatmap tensors.
                Each tensor shape: (batch_size, channels, height, width).
            normalize (bool): Normalize heatmaps to [0, 1] before saving.
            overview (bool): Save an overview grid image of heatmaps per layer.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)

        total_saved = 0

        for layer_name, heatmap_tensor in overlays_dict.items():
            layer_path = save_dir / layer_name
            layer_path.mkdir(parents=True, exist_ok=True)

            N, C, H, W = heatmap_tensor.shape  # batch, channel, height, width

            for n in range(N):

                # for a in range(A):
                hmap = heatmap_tensor[n]  # shape (C, H, W)
                if normalize:
                    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-5)
                fmap_byte = (hmap * 255).clamp(0, 255).byte()
                write_png(fmap_byte.cpu(), layer_path / f"sample_{n}.png")
                total_saved += 1

            # Save overview plot per sample
            if overview:
                fig_cols = math.ceil(math.sqrt(N))
                fig_rows = math.ceil(N / fig_cols)
                fig, axes = plt.subplots(
                    fig_rows,
                    fig_cols,
                    figsize=(fig_cols * 2, fig_rows * 2),
                    layout="compressed",
                )
                axes = axes.flatten()

                for n in range(N):
                    hmap = heatmap_tensor[n]
                    if normalize:
                        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-5)
                    axes[n].imshow(hmap.permute(1, 2, 0).cpu(), cmap="gray")
                    axes[n].axis("off")
                    axes[n].set_title(f"Map {n}")

                for n in range(N, len(axes)):
                    axes[n].axis("off")

                fig.suptitle(f"{layer_name} | Sample {n}")
                plt.tight_layout()
                overview_path = layer_path / f"overview_{layer_name}_sample_{n}.png"
                plt.savefig(overview_path, dpi=150)
                plt.close(fig)

                if self.verbose:
                    print(f"[Logger] Overview image for '{layer_name}', sample {n} saved to '{overview_path}'")

        if self.verbose:
            print(f"[Logger] Saved {total_saved} heatmap images to '{save_dir}'")
