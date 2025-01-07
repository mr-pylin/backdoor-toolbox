import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import Logger


class CleanNet:
    def __init__(
        self,
        model: nn.Module,
        config: dict[str,],
        logger: Logger,
        verbose: True,
    ):

        self.seed = config["misc"]["seed"]
        torch.manual_seed(self.seed)

        self.config = config
        self.device = self.config["misc"]["device"]
        self.model = model.to(self.device)
        self.logger = logger
        self.verbose = verbose

    def train_and_validate(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        train_acc_per_epoch = []
        train_loss_per_epoch = []
        val_acc_per_epoch = []
        val_loss_per_epoch = []

        criterion = nn.CrossEntropyLoss()
        optimizer = self.config["train"]["optimizer"](self.model.parameters(), lr=self.config["train"]["initial_lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, threshold=1e-3)

        train_acc_metric = MulticlassAccuracy(num_classes=self.config["model"]["num_classes"]).to(self.device)
        val_acc_metric = MulticlassAccuracy(num_classes=self.config["model"]["num_classes"]).to(self.device)

        train_loader = DataLoader(train_dataset, batch_size=self.config["train"]["train_batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["train"]["val_batch_size"], shuffle=False)

        epochs = self.config["train"]["epochs"]

        self.logger.save_hyperparameters(
            Path("train-val"),
            "hyperparameters",
            epochs=epochs,
            criterion=criterion.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )

        for epoch in range(epochs):
            # train phase
            self.model.train()
            train_loss = 0

            for x, y_true in train_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model(x)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item() * len(x)
                train_acc_metric.update(y_pred, y_true)

            train_loss /= len(train_dataset)
            train_acc = train_acc_metric.compute().item()
            train_acc_metric.reset()

            # validation phase
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true in val_loader:
                    x, y_true = x.to(self.device), y_true.to(self.device)
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y_true)

                    val_loss += loss.item() * len(x)
                    val_acc_metric.update(y_pred, y_true)

            val_loss /= len(val_dataset)
            val_acc = val_acc_metric.compute().item()
            val_acc_metric.reset()

            scheduler.step(val_loss)

            train_acc_per_epoch.append(train_acc)
            train_loss_per_epoch.append(train_loss)
            val_acc_per_epoch.append(val_acc)
            val_loss_per_epoch.append(val_loss)

            # save a report
            self.logger.save_metrics(
                Path("train-val"),
                "report",
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            # save weights and biases
            self.logger.save_weights(
                Path("train-val/weights"),
                f"epoch_{epoch}",
                self.model,
                only_state_dict=True,
            )

            if self.verbose:
                print(
                    f"epoch {epoch+1:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | train[loss: {train_loss:.5f} - acc: {train_acc*100:5.2f}%] | validation[loss: {val_loss:.5f} - acc: {val_acc*100:5.2f}%]"
                )
        # plot
        self.logger.save_plot(
            Path("train-val/plots"),
            "loss",
            "svg",
            "Loss",
            "Loss over time",
            False,
            train_loss=train_loss_per_epoch,
            val_loss=val_loss_per_epoch,
        )
        self.logger.save_plot(
            Path("train-val/plots"),
            "accuracy",
            "svg",
            "Accuracy",
            "Accuracy over time",
            False,
            train_acc=train_acc_per_epoch,
            val_acc=val_acc_per_epoch,
        )

    def test(
        self,
        test_dataset: Dataset,
    ):
        true_labels = []
        predictions = []

        criterion = nn.CrossEntropyLoss()
        test_loader = DataLoader(test_dataset, batch_size=self.config["train"]["test_batch_size"], shuffle=False)
        test_acc_metric = MulticlassAccuracy(num_classes=self.config["model"]["num_classes"]).to(self.device)
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model(x)
                loss = criterion(y_pred, y_true)

                test_loss += loss.item() * len(x)
                test_acc_metric.update(y_pred, y_true)

                true_labels.extend(y_true.cpu())
                predictions.extend(y_pred.argmax(dim=1).cpu())

        test_loss /= len(test_dataset)
        test_acc = test_acc_metric.compute().item()
        test_acc_metric.reset()

        # save a report
        self.logger.save_metrics(
            Path("test"),
            "report",
            test_loss=test_loss,
            test_acc=test_acc,
        )

        # save confusion matrix
        predictions = torch.tensor(predictions).to("cpu")
        true_labels = torch.tensor(true_labels).to("cpu")
        confmat = MulticlassConfusionMatrix(self.config["model"]["num_classes"])
        cm = confmat(predictions, true_labels)
        self.logger.save_confusion_matrix(
            Path("test"),
            "confusion_matrix",
            cm,
            list(range(10)),
        )

        if self.verbose:
            print(f"test[loss: {test_loss:.5f} - acc: {test_acc*100:5.2f}%]")
