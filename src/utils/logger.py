import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class Logger:
    """
    Initializes the Logger with a root directory and optional flags for
    appending data and verbosity.

    Args:
        root (Path): The base directory where all logs will be saved.
        append_data (bool): Whether to append to existing files or create new ones. Default is True.
        verbose (bool): Whether to print detailed information about the saving process. Default is True.
    """

    def __init__(self, root: Path, append_data: bool = True, verbose: bool = True):
        if append_data:
            self.root = Path(f"{root}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}")
        else:
            self.root = root
        self.verbose = verbose

        # make directory if not available
        self.root.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, path: Path, filename: str, **data):
        """
        Initializes the Logger with a root directory and optional flags for
        appending data and verbosity.

        Args:
            root (Path): The base directory where all logs will be saved.
            append_data (bool): Whether to append to existing files or create new ones. Default is True.
            verbose (bool): Whether to print detailed information about the saving process. Default is True.
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

    def save_hyperparameters(self, path: Path, filename: str, **hyperparameters):
        """
        Saves the hyperparameters used in training to a JSON file.

        Args:
            path (Path): The subdirectory where the hyperparameters file will be saved.
            filename (str): The name of the JSON file where hyperparameters will be saved.
            **hyperparameters: Hyperparameters used in the training process, with keys as parameter names
                               and values as the corresponding hyperparameter values.
        """

        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.json"

        with save_path.open("w") as f:
            json.dump(hyperparameters, f, indent=4)

        if self.verbose:
            print(f"[Logger]: Hyperparameters saved to {save_path}")

    def save_confusion_matrix(self, path: Path, filename: str, cm: np.ndarray | torch.Tensor, unique_labels: list):
        """
        Saves the confusion matrix as a CSV file with true and predicted labels as the first row and column.

        Args:
            path (Path): The subdirectory where the confusion matrix file will be saved.
            filename (str): The name of the CSV file where the confusion matrix will be saved.
            cm (np.ndarray | torch.Tensor): The confusion matrix, either as a NumPy array or a PyTorch tensor.
            unique_labels (list): A list of unique labels (classes) to be used as the first row and column.
        """

        save_dir = self.root / path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.csv"

        # Convert torch.Tensor to numpy.ndarray if necessary
        if isinstance(cm, torch.Tensor):
            cm = cm.cpu().numpy()

        # Create the header with predicted labels (first row)
        header = ["True/Pred"] + unique_labels

        # Create the matrix with true labels as the first column
        cm_with_labels = np.vstack([header, np.column_stack([unique_labels, cm])])

        # Convert the matrix to string to avoid dtype issues
        cm_with_labels = cm_with_labels.astype(str)

        # Save the confusion matrix as a CSV
        np.savetxt(save_path, cm_with_labels, delimiter=",", fmt="%s", comments="")

        if self.verbose:
            print(f"[Logger]: Confusion matrix saved to {save_path}")

    def save_plot(self, path: Path, filename: str, save_format: str, ylabel: str, title: str, show: bool = False, **data):
        """
        Saves a training plot (e.g., loss, accuracy) as an image file. Multiple metrics can be plotted on the same figure.

        Args:
            path (Path): The subdirectory where the plot file will be saved.
            filename (str): The name of the plot file (without extension).
            save_format (str): The format in which to save the plot. Must be either 'png' or 'svg'.
            show (bool): Whether to display the plot using plt.show(). Default is False.
            ylabel (str): The label for the y-axis. Default is "Value".
            title (str): The title of the plot. Default is "Training Plot".
            **data: Key-value pairs of data to plot, where the key is the label (e.g., "train_loss")
                    and the value is the data (e.g., list of loss values).

        Raises:
            ValueError: If the save_format is not 'png' or 'svg'.
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

    def save_weights(self, path: Path, filename: str, model: nn.Module, only_state_dict: bool = True):
        """
        Saves the model weights (either the full model or only the state_dict) to a .pth file.

        Args:
            path (Path): The subdirectory where the model weights file will be saved.
            filename (str): The name of the file where the model weights will be saved.
            model (nn.Module): The PyTorch model whose weights are to be saved.
            only_state_dict (bool): If True, only the state_dict of the model is saved. Default is True.
                                    If False, the entire model is saved.
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


if __name__ == "__main__":
    from torch import nn, optim

    log_dir = Path("../../logs/temp")
    logger = Logger(log_dir, append_data=True, verbose=True)

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
        title="Train and Validation loss over epochs",
        ylabel="Loss",
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
        title="Train and Validation CDA over epochs",
        ylabel="CDA",
        show=True,
        train_cda=train_cda,
        val_cda=val_cda,
    )

    # example model (a simple NN)
    model_1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    logger.save_weights(path=Path("train-val/checkpoints"), filename="epoch_1_parameters", model=model_1, only_state_dict=True)
    model_2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    logger.save_weights(path=Path("test/checkpoints"), filename="parameters", model=model_2, only_state_dict=True)
