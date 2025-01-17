from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision.transforms import v2

from backdoor_toolbox.routines.base import BaseRoutine
from backdoor_toolbox.routines.neutral.config import config
from backdoor_toolbox.utils.logger import Logger


class NeutralRoutine(BaseRoutine):
    def __init__(self):

        self.config: dict[str,] = config

        # extract common values from config file
        self.num_classes = self.config["dataset"]["num_classes"]
        self.seed = self.config["misc"]["seed"]
        self.device = self.config["misc"]["device"]
        self.verbose = self.config["misc"]["verbose"]

        # set manual seed
        torch.manual_seed(self.seed)

        # initialize the logger
        self.logger = Logger(
            root=self.config["log"]["root"],
            include_date=self.config["log"]["include_date"],
            verbose=self.verbose,
        )

        # save current config in the log
        self.logger.save_configs(
            self.config["log"]["config"]["path"],
            self.config["log"]["config"]["filename"],
        )

        # future attributes
        self.train_set: Dataset
        self.val_set: Dataset
        self.test_set: Dataset
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: DataLoader
        self.model: nn.Module

    def apply(self) -> None:
        # prepare datasets and dataloaders
        datasets, dataloaders = self._prepare_data(
            module_path=f"{self.config["modules"]["dataset"]["root"]}.{self.config["modules"]["dataset"]["file"]}",
        )
        self.train_set, self.val_set, self.test_set = datasets
        self.train_loader, self.val_loader, self.test_loader = dataloaders

        # prepare model
        self.model = self._prepare_model(
            module_path=f"{self.config["modules"]["model"]["root"]}.{self.config["modules"]["model"]["file"]}",
            module_cls=self.config["modules"]["model"]["class"],
            weights=self.config["model"]["weights"],
        )

        # train and validate
        self._train_and_validate()

        # test
        self._test()

    def _prepare_data(
        self,
        module_path: str,
    ) -> tuple[tuple[Dataset, ...], tuple[DataLoader, ...]]:

        # import dataset class
        dataset_cls = getattr(self._import_package(module_path), self.config["modules"]["dataset"]["class"])

        # initialize train and validation set
        train_set = dataset_cls(
            root=self.config["dataset"]["root"],
            train=self.config["dataset"]["train"],
            transform=self.config["dataset"]["transform"],
            target_transform=self.config["dataset"]["target_transform"],
            download=self.config["dataset"]["download"],
            seed=self.seed,
        )

        # split train set into train and validation set
        train_set, val_set = random_split(train_set, self.config["train_val"]["train_val_ratio"])

        # calculate mean and std per channel add `v2.Normalize` to transforms
        if self.config["dataset"]["normalize"]:
            self.mean_per_channel, self.std_per_channel = self._calculate_train_set_mean_and_std(train_set)
            self.config["dataset"]["transform"].transforms.append(v2.Normalize(self.mean_per_channel, self.std_per_channel))
        else:
            self.mean_per_channel = None
            self.std_per_channel = None

        # initialize test set
        test_set = dataset_cls(
            root=self.config["dataset"]["root"],
            train=not self.config["dataset"]["train"],
            transform=self.config["dataset"]["transform"],
            target_transform=self.config["dataset"]["target_transform"],
            download=self.config["dataset"]["download"],
            seed=self.seed,
        )

        # initialize data loaders
        train_loader = DataLoader(train_set, batch_size=self.config["train_val"]["train_batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.config["train_val"]["val_batch_size"], shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.config["test"]["test_batch_size"], shuffle=False)

        return (train_set, val_set, test_set), (train_loader, val_loader, test_loader)

    def _prepare_model(
        self,
        module_path: str,
        module_cls: str,
        weights: bool | str,
    ) -> nn.Module:

        # import model class
        model_cls = getattr(self._import_package(module_path), module_cls)

        # initialize the model
        model = model_cls(
            in_features=self.config["dataset"]["image_shape"][0],
            num_classes=self.num_classes,
            weights=weights,
            device=self.device,
            verbose=self.verbose,
        )

        return model

    def _train_and_validate(self) -> None:
        train_acc_per_epoch = []
        train_loss_per_epoch = []
        val_acc_per_epoch = []
        val_loss_per_epoch = []

        # initialize criterion, optimizer and lr_scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = self.config["train_val"]["optimizer"](self.model.parameters(), **self.config["train_val"]["optimizer_params"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.config["train_val"]["scheduler_params"])

        # initialize train and validation accuracy metric (same concept as Clean Data Accuracy)
        train_acc_metric = MulticlassAccuracy(self.num_classes).to(self.device)
        val_acc_metric = MulticlassAccuracy(self.num_classes).to(self.device)

        epochs = self.config["train_val"]["epochs"]

        # store hyperparameters as a json file
        self.logger.save_hyperparameters(
            Path(self.config["log"]["hyperparameters"]["path"]),
            self.config["log"]["hyperparameters"]["filename"],
            epochs=epochs,
            mean_per_channel=self.mean_per_channel,
            std_per_channel=self.std_per_channel,
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

        for n, d in [
            ("train", self.train_set),
            ("val", self.val_set),
        ]:
            self.logger.save_demo(
                Path(self.config["log"]["demo"]["train_path"]),
                n,
                self.model,
                d,
                self.config["log"]["demo"]["nrows"],
                self.config["log"]["demo"]["ncols"],
                show=self.config["log"]["demo"]["show"],
                device=self.device,
            )

    def _test(self) -> None:
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
        predictions, true_labels = map(torch.tensor, [predictions, true_labels])

        confmat = MulticlassConfusionMatrix(self.num_classes)
        cm = confmat(predictions, true_labels)

        self.logger.save_confusion_matrix(
            Path(self.config["log"]["confusion_matrix"]["path"]),
            self.config["log"]["confusion_matrix"]["filename"],
            cm,
            list(range(self.num_classes)),
        )

        self.logger.save_demo(
            Path(self.config["log"]["demo"]["test_path"]),
            "test",
            self.model,
            self.test_set,
            self.config["log"]["demo"]["nrows"],
            self.config["log"]["demo"]["ncols"],
            show=self.config["log"]["demo"]["show"],
            device=self.device,
        )
