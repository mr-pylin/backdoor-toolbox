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


# instantiate a logger to save parameters, plots, weights, ...
logger = Logger(root=config["logger"]["root"], verbose=config["misc"]["verbose"])


class NeutralRoutine(BaseRoutine):

    def __init__(self):
        # set manual seed
        torch.manual_seed(config["misc"]["seed"])

        # save current config in the log
        logger.save_configs(
            src_path=Path(config["logger"]["config"]["src_path"]),
            dst_path=Path(config["logger"]["config"]["dst_path"]),
            filename=config["logger"]["config"]["filename"],
        )

    def apply(self) -> None:
        # prepare datasets
        train_set, val_set, test_set = self._prepare_data()

        # prepare model and initialize weights
        model = self._prepare_model()

        # train and validate
        self._train_and_validate(train_set, val_set, model)

        # test
        self._test(test_set, model)

    def _prepare_data(self) -> tuple[Dataset, Dataset, Dataset]:

        # import dataset class
        dataset_cls = getattr(
            self._import_package(f"{config["modules"]["dataset"]["root"]}.{config["modules"]["dataset"]["file"]}"),
            config["modules"]["dataset"]["class"],
        )

        # initialize train and validation set
        train_set = dataset_cls(
            root=config["dataset"]["root"],
            train=True,
            transform=config["dataset"]["transform"],
            target_transform=config["dataset"]["target_transform"],
            download=config["dataset"]["download"],
        )

        # initialize test set
        test_set = dataset_cls(
            root=config["dataset"]["root"],
            train=False,
            transform=config["dataset"]["transform"],
            target_transform=config["dataset"]["target_transform"],
            download=config["dataset"]["download"],
        )

        # split train set into train and validation set
        train_set, val_set = random_split(
            train_set,
            config["train_val"]["train_val_ratio"],
        )

        # normalize (standardize) samples if needed
        if config["dataset"]["normalize"]:
            self.mean_per_channel, self.std_per_channel = self._calculate_train_set_mean_and_std(train_set)
            config["dataset"]["transform"].transforms.append(v2.Normalize(self.mean_per_channel, self.std_per_channel))
        else:
            self.mean_per_channel = None
            self.std_per_channel = None

        return train_set, val_set, test_set

    def _prepare_model(self) -> nn.Module:

        # import model class
        model_cls = getattr(
            self._import_package(f"{config["modules"]["model"]["root"]}.{config["modules"]["model"]["file"]}"),
            config["modules"]["model"]["class"],
        )

        # initialize the model
        model = model_cls(
            arch=config["modules"]["model"]["params"]["arch"],
            in_channels=config["dataset"]["image_shape"][0],
            num_classes=config["dataset"]["num_classes"],
            weights=config["modules"]["model"]["params"]["weights"],
            device=config["misc"]["device"],
            verbose=config["misc"]["verbose"],
        )

        return model

    def _train_and_validate(self, train_set, val_set, model: nn.Module) -> None:
        # save stats per epoch
        train_loss_per_epoch, train_acc_per_epoch = [], []
        val_loss_per_epoch, val_acc_per_epoch = [], []

        # initialize data loaders
        train_loader = DataLoader(train_set, batch_size=config["train_val"]["train_batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config["train_val"]["val_batch_size"], shuffle=False)

        # initialize criterion, optimizer and lr_scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = config["train_val"]["optimizer"](
            model.parameters(),
            **config["train_val"]["optimizer_params"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config["train_val"]["scheduler_params"],
        )

        # initialize train and validation accuracy metric
        train_acc_metric = MulticlassAccuracy(config["dataset"]["num_classes"]).to(config["misc"]["device"])
        val_acc_metric = MulticlassAccuracy(config["dataset"]["num_classes"]).to(config["misc"]["device"])

        epochs = config["train_val"]["epochs"]

        # store hyperparameters as a json file
        logger.save_hyperparameters(
            path=Path(config["logger"]["hyperparameters"]["path"]),
            filename=config["logger"]["hyperparameters"]["filename"],
            epochs=epochs,
            mean_per_channel=self.mean_per_channel,
            std_per_channel=self.std_per_channel,
            criterion=criterion.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )

        for epoch in range(1, epochs + 1):
            # train phase
            model.train()
            train_loss = 0

            for x, y_true in train_loader:
                # move data to <device>
                x, y_true = x.to(config["misc"]["device"]), y_true.to(config["misc"]["device"])

                # forward and backward pass
                y_pred = model(x)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # calculate and store metrics per batch
                train_loss += loss.item() * len(x)
                train_acc_metric.update(y_pred, y_true)

            # calculate metrics per epoch
            train_loss /= len(train_loader.dataset)
            train_acc = train_acc_metric.compute().item()

            # store metrics per epoch
            train_acc_per_epoch.append(train_acc)
            train_loss_per_epoch.append(train_loss)

            # reset metrics for the next epoch
            train_acc_metric.reset()

            # validation phase
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true in val_loader:
                    # move data to <device>
                    x, y_true = x.to(config["misc"]["device"]), y_true.to(config["misc"]["device"])

                    # forward pass
                    y_pred = model(x)
                    loss = criterion(y_pred, y_true)

                    # calculate and store metrics per batch
                    val_loss += loss.item() * len(x)
                    val_acc_metric.update(y_pred, y_true)

            # calculate metrics per epoch
            val_loss /= len(val_loader.dataset)
            val_acc = val_acc_metric.compute().item()

            # store metrics per epoch
            val_acc_per_epoch.append(val_acc)
            val_loss_per_epoch.append(val_loss)

            # reset metrics for the next epoch
            val_acc_metric.reset()

            # lr scheduler step
            scheduler.step(val_loss)

            # print results per epoch in the standard output
            if config["misc"]["verbose"]:
                print(
                    f"[Train]: epoch {epoch:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | "
                    f"train[loss: {train_loss:.5f} - acc: {train_acc*100:5.2f}%] | "
                    f"validation[loss: {val_loss:.5f} - acc: {val_acc*100:5.2f}%]"
                )

            # store metrics as a csv file for each epoch
            logger.save_metrics(
                path=Path(config["logger"]["metrics"]["train_path"]),
                filename=config["logger"]["metrics"]["filename"],
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            # store weights and biases as a .pth file for each epoch
            logger.save_weights(
                path=Path(config["logger"]["weights"]["path"]),
                filename=config["logger"]["weights"]["filename"],
                model=model,
                epoch=epoch,
                only_state_dict=config["logger"]["weights"]["only_state_dict"],
            )

        # store and/or plot train and validation metrics per epoch
        # y_min = min(min(train_acc_per_epoch), min(val_acc_per_epoch))
        # y_max = max(max(train_acc_per_epoch), max(val_acc_per_epoch))
        for metric in config["logger"]["plot_metrics"]["metrics"]:
            if metric["filename"] == "loss":
                data = {"Train": train_loss_per_epoch, "Validation": val_loss_per_epoch}
            elif metric["filename"] == "accuracy":
                data = {"Train": train_acc_per_epoch, "Validation": val_acc_per_epoch}
            else:
                raise ValueError(f"Unknown metric: {metric["filename"]}")

            logger.plot_and_save_metrics(
                path=Path(config["logger"]["plot_metrics"]["path"]),
                filename=metric["filename"],
                save_format=config["logger"]["plot_metrics"]["save_format"],
                ylabel=metric["ylabel"],
                title=metric["title"],
                data=data,
                show=config["logger"]["plot_metrics"]["show"],
                # ylim=(y_min, y_max) if metric["filename"].lower() != "loss" else None,
                ylim=None,
                markers=config["logger"]["plot_metrics"]["markers"],
            )

        # store and/or plot demo images with true and predicted labels
        for data_role, data_value in [("train", train_set), ("val", val_set)]:
            logger.save_image_predictions(
                path=Path(config["logger"]["pred_demo"]["train_path"]),
                filename=data_role,
                model=model,
                dataset=data_value,
                nrows=config["logger"]["pred_demo"]["nrows"],
                ncols=config["logger"]["pred_demo"]["ncols"],
                save_grid=config["logger"]["pred_demo"]["save_grid"],
                show_grid=config["logger"]["pred_demo"]["show_grid"],
                clamp=config["logger"]["pred_demo"]["clamp"],
            )

    def _test(self, test_set, model: nn.Module) -> None:
        # save true/pred labels for confusion matrix
        true_labels = []
        predictions = []

        # initialize data loader
        test_loader = DataLoader(test_set, batch_size=config["test"]["test_batch_size"], shuffle=False)

        # initialize criterion
        criterion = nn.CrossEntropyLoss()

        # initialize test acc (accuracy) metric
        # move the metric to <device>
        test_acc_metric = MulticlassAccuracy(config["dataset"]["num_classes"]).to(config["misc"]["device"])

        # test phase
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for x, y_true in test_loader:
                # move data to <device>
                x, y_true = x.to(config["misc"]["device"]), y_true.to(config["misc"]["device"])

                # forward pass
                y_pred = model(x)
                loss = criterion(y_pred, y_true)

                # calculate and store metrics per batch
                test_loss += loss.item() * len(x)
                test_acc_metric.update(y_pred, y_true)

                # extend true/pred labels
                true_labels.extend(y_true.cpu())
                predictions.extend(y_pred.argmax(dim=1).cpu())

        # calculate and store metrics
        test_loss /= len(test_loader.dataset)
        test_acc = test_acc_metric.compute().item()

        # reset metric
        test_acc_metric.reset()

        # print results in the standard output
        if config["misc"]["verbose"]:
            print(f"[Test]: test[loss: {test_loss:.5f} - acc: {test_acc*100:5.2f}%]")

        # store metrics as a csv file
        logger.save_metrics(
            path=Path(config["logger"]["metrics"]["test_path"]),
            filename=config["logger"]["metrics"]["filename"],
            test_loss=test_loss,
            test_acc=test_acc,
        )

        # store confusion matrix as a csv file
        predictions, true_labels = map(torch.tensor, [predictions, true_labels])
        confmat = MulticlassConfusionMatrix(config["dataset"]["num_classes"])
        cm = confmat(predictions, true_labels)
        logger.save_labeled_matrix(
            path=Path(config["logger"]["confusion_matrix"]["path"]),
            filename=config["logger"]["confusion_matrix"]["filename"],
            matrix=cm,
            row0_col0_title="True/Pred",  # Title for the top-left header cell
            row_labels=list(range(config["dataset"]["num_classes"])),  # Class labels
        )

        # store and/or plot demo images with true and predicted labels
        logger.save_image_predictions(
            path=Path(config["logger"]["pred_demo"]["test_path"]),
            filename="test",
            model=model,
            dataset=test_set,
            nrows=config["logger"]["pred_demo"]["nrows"],
            ncols=config["logger"]["pred_demo"]["ncols"],
            save_grid=config["logger"]["pred_demo"]["save_grid"],
            show_grid=config["logger"]["pred_demo"]["show_grid"],
            clamp=config["logger"]["pred_demo"]["clamp"],
        )
