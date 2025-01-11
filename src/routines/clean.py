import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision import models  # do not remove this line

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import Logger


class CleanRoutine:
    def __init__(self, configs: dict[str,], verbose: bool):

        self.config = configs["clean"]

        # initialize the logger
        self.logger = Logger(
            root=self.config["log"]["root"],
            include_date=self.config["log"]["include_date"],
            verbose=self.config["log"]["verbose"],
        )

        # extract necessary values from config file
        self.num_classes = self.config["dataset"]["num_classes"]
        self.seed = self.config["misc"]["seed"]
        self.device = self.config["misc"]["device"]
        self.verbose = verbose

        # set manual seed
        torch.manual_seed(self.seed)

        # future attributes
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: DataLoader
        self.model: nn.Module

    def apply(self):
        # prepare datasets and dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.__prepare_data_loaders()

        # prepare model
        self.model = self.__prepare_model()

        # train and validate
        self.__train_and_validate()

        # test
        self.__test()

    def __import_package(self, package):
        try:
            module = importlib.import_module(package)
            return module
        except ModuleNotFoundError as e:
            print(f"Error: {e}")
            return None

    def __prepare_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:

        # import dataset class
        module_path = f"{self.config["modules"]["dataset"]["root"]}.{self.config["modules"]["dataset"]["file"]}"
        module_cls = self.config["modules"]["dataset"]["class"]
        dataset_cls = getattr(self.__import_package(module_path), module_cls)

        # initialize train and validation set
        trainset = dataset_cls(
            root=self.config["dataset"]["root"],
            train=self.config["dataset"]["train"],
            image_transform=self.config["dataset"]["transform"],
            image_target_transform=self.config["dataset"]["target_transform"],
            download=self.config["dataset"]["download"],
            seed=self.seed,
        )

        # split train set into train and validation set
        trainset, valset = random_split(trainset, self.config["train"]["train_val_ratio"])

        # initialize test set
        testset = dataset_cls(
            root=self.config["dataset"]["root"],
            train=not self.config["dataset"]["train"],
            image_transform=self.config["dataset"]["transform"],
            image_target_transform=self.config["dataset"]["target_transform"],
            download=self.config["dataset"]["download"],
            seed=self.seed,
        )

        # initialize data loaders
        train_bs, val_bs, test_bs = (
            self.config["train"]["train_batch_size"],
            self.config["train"]["val_batch_size"],
            self.config["test"]["test_batch_size"],
        )

        train_loader = DataLoader(trainset, batch_size=train_bs, shuffle=True)
        val_loader = DataLoader(valset, batch_size=val_bs, shuffle=False)
        test_loader = DataLoader(testset, batch_size=test_bs, shuffle=False)

        return train_loader, val_loader, test_loader

    def __prepare_model(self):

        # import model class
        module_path = f"{self.config["modules"]["model"]["root"]}.{self.config["modules"]["model"]["file"]}"
        module_cls = self.config["modules"]["model"]["class"]
        model_cls = getattr(self.__import_package(module_path), module_cls)

        # initialize the model
        if self.config["model"]["weights"]:
            pretrained_weights = eval(f"models.{self.config["model"]["weights"]}")  # import pretrained weights from `torchvision.models`
        else:
            pretrained_weights = None

        model = model_cls(
            weights=pretrained_weights,
            in_features=self.config["dataset"]["image_shape"][0],
            num_classes=self.num_classes,
        )

        # move the `model` to `device`
        model = model.to(self.device)

        return model

    def __train_and_validate(self):
        train_acc_per_epoch = []
        train_loss_per_epoch = []
        val_acc_per_epoch = []
        val_loss_per_epoch = []

        # initialize criterion, optimizer and lr_scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = self.config["train"]["optimizer"](self.model.parameters(), **self.config["train"]["optimizer_params"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.config["train"]["scheduler_params"])

        # initialize train and validation accuracy metric (same concept as Clean Data Accuracy)
        train_acc_metric = MulticlassAccuracy(self.num_classes).to(self.device)
        val_acc_metric = MulticlassAccuracy(self.num_classes).to(self.device)

        epochs = self.config["train"]["epochs"]

        # store hyperparameters as a json file
        self.logger.save_hyperparameters(
            Path(self.config["log"]["hyperparameters"]["path"]),
            self.config["log"]["hyperparameters"]["filename"],
            epochs=epochs,
            criterion=criterion.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )

        for epoch in range(1, epochs + 1):
            # train phase
            self.model.train()
            train_loss = 0

            for x, y_true in self.train_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model(x)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item() * len(x)
                train_acc_metric.update(y_pred, y_true)

            train_loss /= len(self.train_loader.dataset)
            train_acc = train_acc_metric.compute().item()
            train_acc_per_epoch.append(train_acc)
            train_loss_per_epoch.append(train_loss)
            train_acc_metric.reset()

            # validation phase
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true in self.val_loader:
                    x, y_true = x.to(self.device), y_true.to(self.device)
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y_true)

                    val_loss += loss.item() * len(x)
                    val_acc_metric.update(y_pred, y_true)

            val_loss /= len(self.val_loader.dataset)
            val_acc = val_acc_metric.compute().item()
            val_acc_per_epoch.append(val_acc)
            val_loss_per_epoch.append(val_loss)
            val_acc_metric.reset()

            # lr scheduler step
            scheduler.step(val_loss)

            # print results per epoch in the standard output
            if self.verbose:
                print(
                    f"[Train]: epoch {epoch:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | train[loss: {train_loss:.5f} - acc: {train_acc*100:5.2f}%] | validation[loss: {val_loss:.5f} - acc: {val_acc*100:5.2f}%]"
                )

            # store metrics as a csv file for each epoch
            self.logger.save_metrics(
                Path(self.config["log"]["metrics"]["train_path"]),
                self.config["log"]["metrics"]["filename"],
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            # store weights and biases as a .pth file for each epoch
            self.logger.save_weights(
                Path(self.config["log"]["weights"]["path"]),
                f"epoch_{epoch}",
                self.model,
                only_state_dict=True,
            )

        # store and/or plot train and validation metrics per epoch
        for metric in self.config["log"]["plot"]["metrics"]:
            if metric["filename"] == "loss":
                data = {"train_loss": train_loss_per_epoch, "val_loss": val_loss_per_epoch}
            elif metric["filename"] == "accuracy":
                data = {"train_acc": train_acc_per_epoch, "val_acc": val_acc_per_epoch}
            else:
                raise ValueError(f"Unknown metric: {metric["filename"]}")

            self.logger.save_plot(
                Path(self.config["log"]["plot"]["path"]),
                metric["filename"],
                self.config["log"]["plot"]["save_format"],
                metric["ylabel"],
                metric["title"],
                metric["show"],
                **data,
            )

    def __test(self):
        true_labels = []
        predictions = []

        # initialize criterion
        criterion = nn.CrossEntropyLoss()

        # initialize test accuracy metric (same concept as Clean Data Accuracy)
        test_acc_metric = MulticlassAccuracy(self.num_classes).to(self.device)

        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for x, y_true in self.test_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model(x)
                loss = criterion(y_pred, y_true)

                test_loss += loss.item() * len(x)
                test_acc_metric.update(y_pred, y_true)

                true_labels.extend(y_true.cpu())
                predictions.extend(y_pred.argmax(dim=1).cpu())

        test_loss /= len(self.test_loader.dataset)
        test_acc = test_acc_metric.compute().item()
        test_acc_metric.reset()

        if self.verbose:
            print(f"[Test]: test[loss: {test_loss:.5f} - acc: {test_acc*100:5.2f}%]")

        # store metrics as a csv file
        self.logger.save_metrics(
            Path(self.config["log"]["metrics"]["test_path"]),
            self.config["log"]["metrics"]["filename"],
            test_loss=test_loss,
            test_acc=test_acc,
        )

        # store confusion matrix as a csv file
        predictions = torch.tensor(predictions).to("cpu")
        true_labels = torch.tensor(true_labels).to("cpu")

        confmat = MulticlassConfusionMatrix(self.num_classes)
        cm = confmat(predictions, true_labels)

        self.logger.save_confusion_matrix(
            Path(self.config["log"]["confusion_matrix"]["path"]),
            self.config["log"]["confusion_matrix"]["filename"],
            cm,
            list(range(self.num_classes)),
        )
