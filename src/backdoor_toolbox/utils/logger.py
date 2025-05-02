import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class Logger:
    """
    A logging utility for saving configurations, hyperparameters, metrics, confusion matrices, plots, and model weights.

    Attributes:
        root (Path): Root directory for saving logs.
        verbose (bool): Flag to print log messages during saving.
    """

    def __init__(self, root: Path, include_date: bool = True, verbose: bool = True):
        """
        Initializes the Logger object.

        Args:
            root (Path): Root directory where logs will be saved.
            include_date (bool): Whether to include a timestamp in the directory name. Defaults to True.
            verbose (bool): Flag to print log messages. Defaults to True.
        """
        if include_date:
            self.root = Path(f"{root}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}")
        else:
            self.root = root
        self.verbose = verbose

        # make directory if not available
        self.root.mkdir(parents=True, exist_ok=True)

    def save_configs(self, src_path: Path, filename: str) -> None:
        """
        Saves a copy of the configuration file to the log directory.

        Args:
            src_path (Path): Path to the source configuration file.
            filename (str): Name of the configuration file to save.
        """
        save_dir = self.root
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.py"

        with open(f"{src_path}/{filename}.py", "rb") as src:
            with open(save_path, "wb") as dst:
                dst.write(src.read())

        if self.verbose:
            print(f"[Logger]: A copy of {filename}.py saved to {save_path}")

    def save_hyperparameters(self, path: Path, filename: str, **hyperparameters) -> None:
        """
        Saves the hyperparameters as a JSON file.

        Args:
            path (Path): Subdirectory where the hyperparameters will be saved.
            filename (str): Name of the file to save.
            hyperparameters (Dict[str, Union[str, int, float]]): Hyperparameters to save.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.json"

        with save_path.open("w") as f:
            json.dump(hyperparameters, f, indent=4)

        if self.verbose:
            print(f"[Logger]: Hyperparameters saved to {save_path}")

    def save_metrics(self, path: Path, filename: str, **data) -> None:
        """
        Saves metrics as a CSV file.

        Args:
            path (Path): Subdirectory where the metrics will be saved.
            filename (str): Name of the file to save.
            data (Dict[str, Union[str, float, int]]): Metrics to save.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.csv"

        path_exist = save_path.exists()
        with save_path.open("a", newline="") as f:
            writer = csv.writer(f)

            # check if the file exists to decide whether to write a header
            if not path_exist:
                writer.writerow(data.keys())

            writer.writerow(data.values())

        if self.verbose:
            print(f"[Logger]: Metrics appended to {save_path}")

    def save_confusion_matrix(
        self,
        path: Path,
        filename: str,
        cm: np.ndarray | torch.Tensor,
        cm_title: str,
        unique_labels: list,
    ) -> None:
        """
        Saves the confusion matrix as a CSV file.

        Args:
            path (Path): Subdirectory where the confusion matrix will be saved.
            filename (str): Name of the file to save.
            cm (np.ndarray | torch.Tensor): Confusion matrix to save.
            unique_labels (list[str]): List of unique labels to include as header.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.csv"

        # Convert torch.Tensor to numpy.ndarray if necessary
        if isinstance(cm, torch.Tensor):
            cm = cm.cpu().numpy()

        # Create the header with predicted labels (first row)
        header = [cm_title] + unique_labels

        # Create the matrix with true labels as the first column
        cm_with_labels = np.vstack([header, np.column_stack([unique_labels, cm])])

        # Convert the matrix to string to avoid dtype issues
        cm_with_labels = cm_with_labels.astype(str)

        # Save the confusion matrix as a CSV
        np.savetxt(save_path, cm_with_labels, delimiter=",", fmt="%s", comments="")

        if self.verbose:
            print(f"[Logger]: Confusion matrix saved to {save_path}")

    def save_plot(
        self, path: Path, filename: str, save_format: str, ylabel: str, title: str, show: bool = False, **data
    ) -> None:
        """
        Saves a plot of the provided data.

        Args:
            path (Path): Subdirectory where the plot will be saved.
            filename (str): Name of the file to save.
            save_format (str): Format for saving the plot. Can be 'png' or 'svg'.
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.
            show (bool): Whether to show the plot after saving. Defaults to False.
            data (Dict[str, list[Union[int, float]]]): Data series to plot.
        """
        if save_format not in ["png", "svg"]:
            raise ValueError("Invalid save_format. Only 'png' and 'svg' are supported.")

        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.{save_format}"

        # create the plot
        plt.figure(figsize=(12, 9), layout="compressed")

        # plot each data series
        for label, values in data.items():
            plt.plot(values, label=label)

        # add labels, title, and legend
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(range(1, len(values) + 1))
        plt.legend(loc="best")

        # save the plot
        plt.savefig(save_path, format=save_format, bbox_inches="tight")

        # show the plot if specified
        if show:
            plt.show()

        # close the plot to avoid memory issues in long training runs
        plt.close()

        if self.verbose:
            print(f"[Logger]: Plot saved to {save_path}")

    def save_weights(self, path: Path, filename: str, model: nn.Module, only_state_dict: bool = True) -> None:
        """
        Saves the model weights.

        Args:
            path (Path): Subdirectory where the model weights will be saved.
            filename (str): Name of the file to save.
            model (nn.Module): The model whose weights are to be saved.
            only_state_dict (bool): Whether to save only the state_dict. Defaults to True.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.pth"

        if only_state_dict:
            torch.save(model.state_dict(), save_path)
        else:
            torch.save(model, save_path)

        if self.verbose:
            print(f"[Logger]: Model weights saved to {save_path}")

    def save_demo(self, path, filename, model, dataset, nrows, ncols, show, device, clamp=True) -> None:
        """
        Saves a demo image of the model predictions.

        Args:
            path (Path): Subdirectory where the demo will be saved.
            filename (str): Name of the file to save.
            model (nn.Module): The model to evaluate.
            dataset: Dataset to sample demo images from.
            nrows (int): Number of rows in the demo plot.
            ncols (int): Number of columns in the demo plot.
            show (bool): Whether to display the demo plot after saving.
            device (torch.device): Device to run the model on.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.png"

        data = next(iter(DataLoader(dataset, batch_size=nrows * ncols, shuffle=False)))

        labels = data[1].to("cpu")
        model.to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(data[0].to(device)).argmax(dim=1).cpu()

        # plot
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5), layout="constrained")
        plt.suptitle(f"First {nrows * ncols} images of {filename}")
        for i in range(nrows):
            for j in range(ncols):
                if clamp and torch.is_floating_point(data[0][i * ncols + j]):
                    img = torch.clamp(data[0][i * ncols + j], 0, 1)
                axs[i, j].imshow(img.permute(1, 2, 0), cmap="gray")
                axs[i, j].set_title(f"t:{labels[i * ncols + j].item()},p:{predictions[i * ncols + j].item()}")
                axs[i, j].axis("off")

        # save the plot
        plt.savefig(save_path, format="png", bbox_inches="tight")

        if show:
            plt.show()

        # close the plot to avoid memory issues in long training runs
        plt.close()

        if self.verbose:
            print(f"[Logger]: Demo saved to {save_path}")

    def save_trigger(self, path, filename, trigger_policy, bg_size, bg_color, show) -> None:
        """
        Args:
            path (Path): Subdirectory where the demo will be saved.
            filename (str): Name of the file to save.
            show (bool): Whether to display the demo plot after saving.
        """
        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.png"

        bg = torch.full(size=bg_size, fill_value=bg_color, dtype=torch.float32)
        poisoned_bg = trigger_policy(bg)

        # plot
        plt.figure(figsize=(9, 6), layout="compressed")
        plt.imshow(poisoned_bg.permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.title(trigger_policy.name)

        # save the plot
        plt.savefig(save_path, format="PNG", bbox_inches="tight")

        # show the plot if specified
        if show:
            plt.show()

        # close the plot to avoid memory issues in long training runs
        plt.close()

        if self.verbose:
            print(f"[Logger]: Demo Trigger saved to {save_path}")


if __name__ == "__main__":
    from torch import nn, optim

    log_dir = Path("./logs/temp")
    logger = Logger(log_dir, include_date=True, verbose=True)

    # save metrics in csv file
    metrics_1 = {
        "epoch": 1,
        "lr": 0.001,
        "train_loss": 0.5,
        "val_loss": 0.45,
        "train_cda": 0.87,
        "val_cda": 0.17,
        "train_asr": 0.47,
        "val_asr": 0.37,
    }
    logger.save_metrics(path=Path("train-val"), filename="report", **metrics_1)

    metrics_2 = {
        "epoch": 2,
        "lr": 0.1,
        "train_loss": 0.55,
        "val_loss": 0.65,
        "train_cda": 0.57,
        "val_cda": 0.16,
        "train_asr": 0.27,
        "val_asr": 0.3356,
    }
    logger.save_metrics(path=Path("train-val"), filename="report", **metrics_2)

    metrics_3 = {
        "epoch": 3,
        "lr": 0.01,
        "train_loss": 0.51,
        "val_loss": 0.25,
        "train_cda": 0.83,
        "val_cda": 0.47,
        "train_asr": 0.411,
        "val_asr": 0.327,
    }
    logger.save_metrics(path=Path("train-val"), filename="report", **metrics_3)

    metrics_4 = {
        "test_loss": 0.51,
        "test_cda": 0.83,
        "test_asr": 0.411,
    }
    logger.save_metrics(path=Path("test"), filename="report", **metrics_4)

    # example hyperparameters to save
    optimizer = optim.SGD(params=[nn.Parameter()], lr=0.1)
    hyperparameters = {"batch_size": 32, "optimizer": optimizer.state_dict()}
    logger.save_hyperparameters(path=Path("hyperparameters"), filename="training_params", **hyperparameters)

    # example confusion matrix
    cm = np.array([[50, 10, 0], [5, 35, 5], [15, 5, 45]])
    unique_labels = ["Airplane", "Cat", "Star"]
    logger.save_confusion_matrix(path=Path("test"), filename="confusion_matrix", cm=cm, unique_labels=unique_labels)

    # Example plot data
    train_loss = [0.6, 0.5, 0.4, 0.3, 0.2]
    val_loss = [0.55, 0.45, 0.40, 0.35, 0.30]
    logger.save_plot(
        path=Path("plots"),
        filename="loss_plot_per_epoch",
        save_format="png",
        ylabel="Loss",
        title="Train and Validation loss over epochs",
        show=True,
        train_loss=train_loss,
        val_loss=val_loss,
    )

    train_cda = [0.65, 0.9, 0.47, 0.23, 0.4]
    val_cda = [0.55, 0.34, 0.55, 0.45, 0.21]
    logger.save_plot(
        path=Path("plots"),
        filename="cda_plot_per_epoch",
        save_format="svg",
        ylabel="CDA",
        title="Train and Validation CDA over epochs",
        show=True,
        train_cda=train_cda,
        val_cda=val_cda,
    )

    # example model (a simple NN)
    model_1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    logger.save_weights(
        path=Path("train-val/checkpoints"), filename="epoch_1_parameters", model=model_1, only_state_dict=True
    )
    model_2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    logger.save_weights(path=Path("test/checkpoints"), filename="parameters", model=model_2, only_state_dict=True)
