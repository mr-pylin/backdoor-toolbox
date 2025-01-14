import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision import models  # do not remove this line

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import Logger
from utils.metrics import AttackSuccessRate, CleanDataAccuracy


class QuantizationAttackRoutine:
    def __init__(self, configs: dict[str,], verbose: bool):

        self.config = configs["attack"]

        # initialize the logger
        self.logger = Logger(
            root=self.config["log"]["root"],
            include_date=self.config["log"]["include_date"],
            verbose=self.config["log"]["verbose"],
        )

        # extract necessary values from config file
        self.target_index = self.config["dataset"]["target_index"]
        self.victim_indices = self.config["dataset"]["victim_indices"]
        self.num_classes = self.config["dataset"]["num_classes"]
        self.seed = self.config["misc"]["seed"]
        self.device = self.config["misc"]["device"]
        self.verbose = verbose

        # set manual seed
        torch.manual_seed(self.seed)

        # future attributes
        self.trainset: Dataset
        self.valset: Dataset
        self.testset_asr: Dataset
        self.testset_cda: Dataset
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader_asr: DataLoader
        self.test_loader_cda: DataLoader
        self.model: nn.Module

    def apply(self):
        # prepare datasets and dataloaders
        self.train_loader, self.val_loader, self.test_loader_asr, self.test_loader_cda = self.__prepare_data_loaders()

        # prepare model
        self.model = self.__prepare_model()

        # step 1: backdoor injection (train a backdoored model)

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

        # import dataset classes
        module_path = f"{self.config["modules"]["dataset"]["root"]}.{self.config["modules"]["dataset"]["file"]}"
        clean_dataset_cls = getattr(self.__import_package(module_path), self.config["modules"]["dataset"]["clean_class"])
        poisoned_dataset_cls = getattr(self.__import_package(module_path), self.config["modules"]["dataset"]["poisoned_class"])

        # initialize train and validation set
        self.trainset = poisoned_dataset_cls(
            root=self.config["dataset"]["root"],
            train=self.config["dataset"]["train"],
            clean_transform=self.config["dataset"]["clean_transform"],
            clean_target_transform=self.config["dataset"]["clean_target_transform"],
            download=self.config["dataset"]["download"],
            target_index=self.config["dataset"]["target_index"],
            victim_indices=self.config["dataset"]["victim_indices"],
            poison_ratio=self.config["dataset"]["poison_ratio"],
            poisoned_transform=self.config["dataset"]["poisoned_transform"],
            poisoned_target_transform=self.config["dataset"]["poisoned_target_transform"],
            skip_target_samples=False,
            seed=self.seed,
        )

        self.trainset, self.valset = random_split(
            self.trainset,
            self.config["train"]["train_val_ratio"],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # initialize test sets
        self.testset_asr = poisoned_dataset_cls(
            root=self.config["dataset"]["root"],
            train=not self.config["dataset"]["train"],
            clean_transform=self.config["dataset"]["clean_transform"],
            clean_target_transform=self.config["dataset"]["clean_target_transform"],
            poisoned_transform=self.config["dataset"]["poisoned_transform"],
            poisoned_target_transform=self.config["dataset"]["poisoned_target_transform"],
            download=self.config["dataset"]["download"],
            target_index=self.config["dataset"]["target_index"],
            victim_indices=self.config["dataset"]["victim_indices"],
            poison_ratio=1.0,
            skip_target_samples=True,
            seed=self.seed,
        )

        self.testset_cda = clean_dataset_cls(
            root=self.config["dataset"]["root"],
            train=not self.config["dataset"]["train"],
            image_transform=self.config["dataset"]["clean_transform"],
            image_target_transform=self.config["dataset"]["clean_target_transform"],
            download=self.config["dataset"]["download"],
            seed=self.seed,
        )

        # initialize data loaders
        train_bs, val_bs, test_bs = (
            self.config["train"]["train_batch_size"],
            self.config["train"]["val_batch_size"],
            self.config["test"]["test_batch_size"],
        )

        train_loader = DataLoader(self.trainset, batch_size=train_bs, shuffle=True)
        val_loader = DataLoader(self.valset, batch_size=val_bs, shuffle=False)
        test_loader_asr = DataLoader(self.testset_asr, batch_size=test_bs, shuffle=False)
        test_loader_cda = DataLoader(self.testset_cda, batch_size=test_bs, shuffle=False)

        return train_loader, val_loader, test_loader_asr, test_loader_cda

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
        train_asr_per_epoch = []
        train_cda_per_epoch = []
        train_loss_per_epoch = []
        val_asr_per_epoch = []
        val_cda_per_epoch = []
        val_loss_per_epoch = []

        # initialize criterion, optimizer and lr_scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = self.config["train"]["optimizer"](self.model.parameters(), **self.config["train"]["optimizer_params"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.config["train"]["scheduler_params"])

        # initialize train and validation accuracy metric (same concept as Clean Data Accuracy)
        train_asr_metric = AttackSuccessRate().to(self.device)
        train_cda_metric = CleanDataAccuracy().to(self.device)
        val_asr_metric = AttackSuccessRate().to(self.device)
        val_cda_metric = CleanDataAccuracy().to(self.device)

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

            for x, y_true, poisoned_mask, _ in self.train_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)

                y_pred = self.model(x)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item() * len(x)

                with torch.no_grad():
                    train_asr_metric.update(self.model(x), y_true, poisoned_mask)
                    train_cda_metric.update(self.model(x), y_true, ~poisoned_mask)

            train_loss /= len(self.train_loader.dataset)
            train_asr = train_asr_metric.compute().item()
            train_cda = train_cda_metric.compute().item()
            train_asr_per_epoch.append(train_asr)
            train_cda_per_epoch.append(train_cda)
            train_loss_per_epoch.append(train_loss)
            train_asr_metric.reset()
            train_cda_metric.reset()

            # validation phase
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true, poisoned_mask, _ in self.val_loader:
                    x, y_true = x.to(self.device), y_true.to(self.device)
                    loss = criterion(self.model(x), y_true)
                    val_loss += loss.item() * len(x)

                    val_asr_metric.update(self.model(x), y_true, poisoned_mask)
                    val_cda_metric.update(self.model(x), y_true, ~poisoned_mask)

            val_loss /= len(self.val_loader.dataset)
            val_asr = val_asr_metric.compute().item()
            val_cda = val_cda_metric.compute().item()
            val_asr_per_epoch.append(val_asr)
            val_cda_per_epoch.append(val_cda)
            val_loss_per_epoch.append(val_loss)
            val_asr_metric.reset()
            val_cda_metric.reset()

            # lr scheduler step
            scheduler.step(val_loss)

            # print results per epoch in the standard output
            if self.verbose:
                print(
                    f"[Train]: epoch {epoch:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | train[loss: {train_loss:.5f} - asr: {train_asr*100:5.2f}% - cda: {train_cda*100:5.2f}%] | validation[loss: {val_loss:.5f} - asr: {val_asr*100:5.2f}% - cda: {val_cda*100:5.2f}%]"
                )

            # store metrics as a csv file for each epoch
            self.logger.save_metrics(
                Path(self.config["log"]["metrics"]["train_path"]),
                self.config["log"]["metrics"]["filename"],
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                train_loss=train_loss,
                train_asr=train_asr,
                train_cda=train_cda,
                val_loss=val_loss,
                val_asr=val_asr,
                val_cda=val_cda,
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
            elif metric["filename"] == "asr":
                data = {"train_asr": train_asr_per_epoch, "val_asr": val_asr_per_epoch}
            elif metric["filename"] == "cda":
                data = {"train_cda": train_cda_per_epoch, "val_cda": val_cda_per_epoch}
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
            ("train", self.trainset),
            ("val", self.valset),
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

    def __test(self):
        true_labels_asr, true_labels_cda = [], []
        predictions_asr, predictions_cda = [], []

        # initialize criterion
        criterion = nn.CrossEntropyLoss()

        # initialize test ASR and CDA
        test_asr_metric = AttackSuccessRate().to(self.device)
        test_cda_metric = CleanDataAccuracy().to(self.device)

        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for (x_asr, y_asr_true, poisoned_mask, y_raw), (x_cda, y_cda_true) in zip(self.test_loader_asr, self.test_loader_cda):

                # attack success rate
                x_asr, y_asr_true = x_asr.to(self.device), y_asr_true.to(self.device)
                y_asr_pred = self.model(x_asr)
                test_asr_metric.update(y_asr_pred, y_asr_true, poisoned_mask)
                true_labels_asr.extend(y_raw.cpu())  # y_raw instead of y_asr_true for better demonstration in confusion matrix
                predictions_asr.extend(y_asr_pred.argmax(dim=1).cpu())

                # clean data accuracy
                x_cda, y_cda_true = x_cda.to(self.device), y_cda_true.to(self.device)
                y_cda_pred = self.model(x_cda)
                test_cda_metric.update(y_cda_pred, y_cda_true, None)
                true_labels_cda.extend(y_cda_true.cpu())
                predictions_cda.extend(y_cda_pred.argmax(dim=1).cpu())

                # total loss of both datasets
                loss = criterion(y_asr_pred, y_asr_true) + criterion(y_cda_pred, y_cda_true)
                test_loss += loss.item() * (len(x_asr) + len(x_cda))

        test_loss /= len(self.test_loader_asr.dataset) + len(self.test_loader_cda.dataset)
        test_asr = test_asr_metric.compute().item()
        test_cda = test_cda_metric.compute().item()
        test_asr_metric.reset()
        test_cda_metric.reset()

        if self.verbose:
            print(f"[Test]: test[loss: {test_loss:.5f} - asr: {test_asr*100:5.2f}% - cda: {test_cda*100:5.2f}%]")

        # store metrics as a csv file
        self.logger.save_metrics(
            Path(self.config["log"]["metrics"]["test_path"]),
            self.config["log"]["metrics"]["filename"],
            test_loss=test_loss,
            test_asr=test_asr,
            test_cda=test_cda,
        )

        # store confusion matrix as a csv file
        true_labels_asr, true_labels_cda, predictions_asr, predictions_cda = map(
            torch.tensor, [true_labels_asr, true_labels_cda, predictions_asr, predictions_cda]
        )

        for n, labels in [
            ("test_asr", (predictions_asr, true_labels_asr)),
            ("test_cda", (predictions_cda, true_labels_cda)),
        ]:
            confmat = MulticlassConfusionMatrix(self.num_classes)
            cm = confmat(*labels)

            self.logger.save_confusion_matrix(
                Path(self.config["log"]["confusion_matrix"]["path"]),
                f"{n}_{self.config["log"]["confusion_matrix"]["filename"]}",
                cm,
                list(range(self.num_classes)),
            )

        for n, d in [
            ("test_asr", self.testset_asr),
            ("test_cda", self.testset_cda),
        ]:
            self.logger.save_demo(
                Path(self.config["log"]["demo"]["test_path"]),
                n,
                self.model,
                d,
                self.config["log"]["demo"]["nrows"],
                self.config["log"]["demo"]["ncols"],
                show=self.config["log"]["demo"]["show"],
                device=self.device,
            )
