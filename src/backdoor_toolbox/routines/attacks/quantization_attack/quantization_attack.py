import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision.transforms import v2

from backdoor_toolbox.routines.attacks.quantization_attack.config import config
from backdoor_toolbox.routines.base import BaseRoutine
from backdoor_toolbox.utils.logger import Logger
from backdoor_toolbox.utils.metrics import AttackSuccessRate, CleanDataAccuracy
from backdoor_toolbox.utils.stats import calculate_mean_and_std

# global configuration file
config: dict[str,] = config

# global logger
logger = Logger(
    root=config["log"]["root"],
    include_date=config["log"]["include_date"],
    verbose=config["misc"]["verbose"],
)


class QuantizationAttackRoutine(BaseRoutine):
    def __init__(self):

        # extract common values from config file
        self.target_index = config["dataset"]["target_index"]
        self.victim_indices = config["dataset"]["victim_indices"]
        self.num_classes = config["dataset"]["num_classes"]
        self.seed = config["misc"]["seed"]
        self.device = config["misc"]["device"]
        self.verbose = config["misc"]["verbose"]

        # set manual seed
        torch.manual_seed(self.seed)

        # save current config in the log
        logger.save_configs(
            config["log"]["config"]["path"],
            config["log"]["config"]["filename"],
        )

        # future attributes
        self.model: nn.Module

    def apply(self) -> None:
        # prepare datasets and dataloaders
        datasets, dataloaders = self._prepare_data(
            module_path=f"{config["modules"]["dataset"]["root"]}.{config["modules"]["dataset"]["file"]}"
        )
        train_set_step_1, val_set_step_1, train_set_step_2, val_set_step_2, test_set_asr, test_set_cda = datasets
        (
            train_loader_step_1,
            val_loader_step_1,
            train_loader_step_2,
            val_loader_step_2,
            test_loader_asr,
            test_loader_cda,
        ) = dataloaders

        # prepare model
        self.model = self._prepare_models(
            module_path=f"{config["modules"]["model"]["root"]}.{config["modules"]["model"]["file"]}",
            module_cls=config["modules"]["model"]["class"],
            weights=config["modules"]["model"]["params"]["weights"],
        )

        # step 1 [backdoor injection]
        backdoor_injection = BackdoorInjection(
            data_sets=(train_set_step_1, val_set_step_1, test_set_asr, test_set_cda),
            data_loaders=(train_loader_step_1, val_loader_step_1, test_loader_asr, test_loader_cda),
            model_bd=self.model,
            normalization_params=(self.mean_per_channel, self.std_per_channel),
        )
        backdoor_injection._train_and_validate()
        backdoor_injection._test()

        # initialize step 2 [backdoor removal]
        backdoor_removal = BackdoorRemoval(
            data_sets=(train_set_step_2, val_set_step_2, test_set_asr, test_set_cda),
            data_loaders=(train_loader_step_2, val_loader_step_2, test_loader_asr, test_loader_cda),
            model_bd=self.model,
            normalization_params=(self.mean_per_channel, self.std_per_channel),
        )
        backdoor_removal._train_and_validate()
        # backdoor_removal._test()

    def _prepare_data(
        self,
        module_path: str,
    ) -> tuple[tuple[Dataset, ...], tuple[DataLoader, ...]]:

        # import dataset classes
        clean_dataset_cls = getattr(self._import_package(module_path), config["modules"]["dataset"]["clean_class"])
        poisoned_dataset_cls = getattr(
            self._import_package(module_path), config["modules"]["dataset"]["poisoned_class"]
        )

        # initialize train and validation set
        train_set_step_1 = poisoned_dataset_cls(
            root=config["dataset"]["root"],
            train=config["dataset"]["train"],
            clean_transform=config["dataset"]["clean_transform"],
            clean_target_transform=config["dataset"]["clean_target_transform"],
            download=config["dataset"]["download"],
            target_index=config["dataset"]["target_index"],
            victim_indices=config["dataset"]["victim_indices"],
            poison_ratio=config["dataset"]["poison_ratio"],
            poisoned_transform=config["dataset"]["poisoned_transform"],
            poisoned_target_transform=config["dataset"]["poisoned_target_transform"],
            skip_target_samples=False,
            seed=self.seed,
        )

        train_set_step_1, val_set_step_1 = random_split(train_set_step_1, config["train_val"]["train_val_ratio"])

        # calculate mean and std per channel add `v2.Normalize` to transforms
        if config["dataset"]["normalize"]:
            self.mean_per_channel, self.std_per_channel = calculate_mean_and_std(train_set_step_1)
            config["dataset"]["clean_transform"].transforms.append(
                v2.Normalize(self.mean_per_channel, self.std_per_channel)
            )
            config["dataset"]["poisoned_transform"].transforms.append(
                v2.Normalize(self.mean_per_channel, self.std_per_channel)
            )
        else:
            self.mean_per_channel = None
            self.std_per_channel = None

        # train_set and val_set for step_2 [ignore label flip]
        train_set_step_2 = copy.deepcopy(train_set_step_1)
        train_set_step_2.dataset.poisoned_target_transform = None
        val_set_step_2 = copy.deepcopy(val_set_step_1)
        val_set_step_2.dataset.poisoned_target_transform = None

        # initialize test sets
        test_set_asr = poisoned_dataset_cls(
            root=config["dataset"]["root"],
            train=not config["dataset"]["train"],
            clean_transform=config["dataset"]["clean_transform"],
            clean_target_transform=config["dataset"]["clean_target_transform"],
            poisoned_transform=config["dataset"]["poisoned_transform"],
            poisoned_target_transform=config["dataset"]["poisoned_target_transform"],
            download=config["dataset"]["download"],
            target_index=config["dataset"]["target_index"],
            victim_indices=config["dataset"]["victim_indices"],
            poison_ratio=1.0,
            skip_target_samples=True,
            seed=self.seed,
        )

        test_set_cda = clean_dataset_cls(
            root=config["dataset"]["root"],
            train=not config["dataset"]["train"],
            transform=config["dataset"]["clean_transform"],
            target_transform=config["dataset"]["clean_target_transform"],
            download=config["dataset"]["download"],
            seed=self.seed,
        )

        # initialize data loaders
        train_loader_step_1 = DataLoader(
            train_set_step_1, batch_size=config["train_val"]["train_batch_size"], shuffle=True
        )
        val_loader_step_1 = DataLoader(val_set_step_1, batch_size=config["train_val"]["val_batch_size"], shuffle=False)
        train_loader_step_2 = DataLoader(
            train_set_step_2, batch_size=config["train_val"]["train_batch_size"], shuffle=True
        )
        val_loader_step_2 = DataLoader(val_set_step_2, batch_size=config["train_val"]["val_batch_size"], shuffle=False)
        test_loader_asr = DataLoader(test_set_asr, batch_size=config["test"]["test_batch_size"], shuffle=False)
        test_loader_cda = DataLoader(test_set_cda, batch_size=config["test"]["test_batch_size"], shuffle=False)

        return (
            train_set_step_1,
            val_set_step_1,
            train_set_step_2,
            val_set_step_2,
            test_set_asr,
            test_set_cda,
        ), (
            train_loader_step_1,
            val_loader_step_1,
            train_loader_step_2,
            val_loader_step_2,
            test_loader_asr,
            test_loader_cda,
        )

    def _prepare_models(
        self,
        module_path: str,
        module_cls: str,
        weights: bool | str,
    ) -> nn.Module:

        # import model class
        model_cls = getattr(self._import_package(module_path), module_cls)

        model = model_cls(
            in_channels=config["dataset"]["image_shape"][0],
            num_classes=self.num_classes,
            weights=weights,
            device=self.device,
            verbose=self.verbose,
            **config["modules"]["model"]["params"]["kwargs"],
        )

        return model


class BackdoorInjection:
    def __init__(
        self,
        data_sets: tuple[Dataset, Dataset, Dataset, Dataset],
        data_loaders: tuple[DataLoader, DataLoader, DataLoader, DataLoader],
        model_bd: nn.Module,
        normalization_params: tuple[list[float], list[float]],
    ):
        self.train_set, self.val_set, self.test_set_asr, self.test_set_cda = data_sets
        self.train_loader, self.val_loader, self.test_loader_asr, self.test_loader_cda = data_loaders
        self.model_bd = model_bd
        self.mean_per_channel, self.std_per_channel = normalization_params

        # extract common values from config file
        self.target_index = config["dataset"]["target_index"]
        self.num_classes = config["dataset"]["num_classes"]
        self.device = config["misc"]["device"]
        self.verbose = config["misc"]["verbose"]

    def _train_and_validate(self):
        train_asr_per_epoch = []
        train_cda_per_epoch = []
        train_loss_per_epoch = []
        val_asr_per_epoch = []
        val_cda_per_epoch = []
        val_loss_per_epoch = []

        # initialize criterion, optimizer and lr_scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = config["train_val"]["optimizer"](
            self.model_bd.parameters(), **config["train_val"]["optimizer_params"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["train_val"]["scheduler_params"])

        # initialize train and validation accuracy metric (same concept as Clean Data Accuracy)
        train_asr_metric = AttackSuccessRate(self.target_index).to(self.device)
        train_cda_metric = CleanDataAccuracy().to(self.device)
        val_asr_metric = AttackSuccessRate(self.target_index).to(self.device)
        val_cda_metric = CleanDataAccuracy().to(self.device)

        epochs = config["train_val"]["step_1_epochs"]

        # store hyperparameters as a json file
        logger.save_hyperparameters(
            Path(f"step_1/{config["log"]["hyperparameters"]["path"]}"),
            config["log"]["hyperparameters"]["filename"],
            epochs=epochs,
            mean_per_channel=self.mean_per_channel,
            std_per_channel=self.std_per_channel,
            criterion=criterion.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )

        for epoch in range(1, epochs + 1):
            # train phase
            self.model_bd.train()
            train_loss = 0

            for x, y_true, poisoned_mask, _ in self.train_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model_bd(x)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item() * len(x)

                with torch.no_grad():
                    train_asr_metric.update(self.model_bd(x), y_true, poisoned_mask)
                    train_cda_metric.update(self.model_bd(x), y_true, ~poisoned_mask)

            train_loss /= len(self.train_loader.dataset)
            train_asr = train_asr_metric.compute().item()
            train_cda = train_cda_metric.compute().item()
            train_asr_per_epoch.append(train_asr)
            train_cda_per_epoch.append(train_cda)
            train_loss_per_epoch.append(train_loss)
            train_asr_metric.reset()
            train_cda_metric.reset()

            # validation phase
            self.model_bd.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true, poisoned_mask, _ in self.val_loader:
                    x, y_true = x.to(self.device), y_true.to(self.device)
                    loss = criterion(self.model_bd(x), y_true)
                    val_loss += loss.item() * len(x)

                    val_asr_metric.update(self.model_bd(x), y_true, poisoned_mask)
                    val_cda_metric.update(self.model_bd(x), y_true, ~poisoned_mask)

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
                    f"[Train_step-1]: epoch {epoch:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | train[loss: {train_loss:.5f} - asr: {train_asr*100:5.2f}% - cda: {train_cda*100:5.2f}%] | validation[loss: {val_loss:.5f} - asr: {val_asr*100:5.2f}% - cda: {val_cda*100:5.2f}%]"
                )

            # store metrics as a csv file for each epoch
            logger.save_metrics(
                Path(f"step_1/{config["log"]["metrics"]["train_path"]}"),
                config["log"]["metrics"]["filename"],
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
            logger.save_weights(
                Path(f"step_1/{config["log"]["weights"]["path"]}"),
                f"epoch_{epoch}",
                self.model_bd,
                only_state_dict=True,
            )

        # store and/or plot train and validation metrics per epoch
        for metric in config["log"]["plot"]["metrics"]:
            if metric["filename"] == "loss":
                data = {"train_loss": train_loss_per_epoch, "val_loss": val_loss_per_epoch}
            elif metric["filename"] == "asr":
                data = {"train_asr": train_asr_per_epoch, "val_asr": val_asr_per_epoch}
            elif metric["filename"] == "cda":
                data = {"train_cda": train_cda_per_epoch, "val_cda": val_cda_per_epoch}
            else:
                raise ValueError(f"Unknown metric: {metric["filename"]}")

            logger.plot_and_save_metrics(
                Path(f"step_1/{config["log"]["plot"]["path"]}"),
                metric["filename"],
                config["log"]["plot"]["save_format"],
                metric["ylabel"],
                metric["title"],
                metric["show"],
                **data,
            )

        for n, d in [
            ("train", self.train_set),
            ("val", self.val_set),
        ]:
            logger.save_image_predictions(
                Path(f"step_1/{config["log"]["demo"]["train_path"]}"),
                n,
                self.model_bd,
                d,
                config["log"]["demo"]["nrows"],
                config["log"]["demo"]["ncols"],
                show=config["log"]["demo"]["show"],
                device=self.device,
            )

    def _test(self):
        true_labels_asr, true_labels_cda = [], []
        predictions_asr, predictions_cda = [], []

        # initialize criterion
        criterion = nn.CrossEntropyLoss()

        # initialize test ASR and CDA
        test_asr_metric = AttackSuccessRate(self.target_index).to(self.device)
        test_cda_metric = CleanDataAccuracy().to(self.device)

        self.model_bd.eval()
        test_loss = 0

        with torch.no_grad():
            for (x_asr, y_asr_true, poisoned_mask, y_raw), (x_cda, y_cda_true) in zip(
                self.test_loader_asr, self.test_loader_cda
            ):

                # attack success rate
                x_asr, y_asr_true = x_asr.to(self.device), y_asr_true.to(self.device)
                y_asr_pred = self.model_bd(x_asr)
                test_asr_metric.update(y_asr_pred, y_asr_true, poisoned_mask)
                true_labels_asr.extend(
                    y_raw.cpu()
                )  # y_raw instead of y_asr_true for better demonstration in confusion matrix
                predictions_asr.extend(y_asr_pred.argmax(dim=1).cpu())

                # clean data accuracy
                x_cda, y_cda_true = x_cda.to(self.device), y_cda_true.to(self.device)
                y_cda_pred = self.model_bd(x_cda)
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
            print(f"[Test_step-1]: test[loss: {test_loss:.5f} - asr: {test_asr*100:5.2f}% - cda: {test_cda*100:5.2f}%]")

        # store metrics as a csv file
        logger.save_metrics(
            Path(f"step_1/{config["log"]["metrics"]["test_path"]}"),
            config["log"]["metrics"]["filename"],
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

            logger.save_labeled_matrix(
                Path(f"step_1/{config["log"]["confusion_matrix"]["path"]}"),
                f"{n}_{config["log"]["confusion_matrix"]["filename"]}",
                cm,
                list(range(self.num_classes)),
            )

        for n, d in [
            ("test_asr", self.test_set_asr),
            ("test_cda", self.test_set_cda),
        ]:
            logger.save_image_predictions(
                Path(f"step_1/{config["log"]["demo"]["test_path"]}"),
                n,
                self.model_bd,
                d,
                config["log"]["demo"]["nrows"],
                config["log"]["demo"]["ncols"],
                show=config["log"]["demo"]["show"],
                device=self.device,
            )


class BackdoorRemoval:
    def __init__(
        self,
        data_sets: tuple[Dataset, Dataset, Dataset, Dataset],
        data_loaders: tuple[DataLoader, DataLoader, DataLoader, DataLoader],
        model_bd: nn.Module,
        normalization_params: tuple[list[float], list[float]],
    ):
        self.train_set, self.val_set, self.test_set_asr, self.test_set_cda = data_sets
        self.train_loader, self.val_loader, self.test_loader_asr, self.test_loader_cda = data_loaders
        self.mean_per_channel, self.std_per_channel = normalization_params

        self.model_bd = model_bd
        self.model_bd_q, self.scaling_factors_bd = self.__quantize_model(self.model_bd)
        self.model_rm = copy.deepcopy(self.model_bd)

        # self.model_bd_q = self.model_bd_q.to(self.device)

        # extract common values from config file
        self.target_index = config["dataset"]["target_index"]
        self.num_classes = config["dataset"]["num_classes"]
        self.device = config["misc"]["device"]
        self.verbose = config["misc"]["verbose"]
        self.epsilon_1 = config["loss"]["epsilon_1"]
        self.epsilon_2 = config["loss"]["epsilon_2"]
        self._lambda = config["loss"]["lambda"]

    def __quantize_model(self, model, scale_bits=8):
        quantized_model = copy.deepcopy(model)
        scaling_factors = []

        for _, param in quantized_model.named_parameters():
            scale = 2 ** (scale_bits - 1) - 1  # Symmetric range for signed values
            max_val = param.detach().abs().max()  # Detach to prevent gradient tracking
            scaling_factor = (
                max_val / scale if max_val > 0 else 1.0
            )  # to prevent producing NaNs or zero during quantization
            scaling_factors.append(scaling_factor)

            # Quantize and dequantize without interfering with gradients
            quantized_param = torch.round(param.detach() / scaling_factor).clamp(-scale, scale) * scaling_factor

            # Update the parameter in the quantized model
            with torch.no_grad():
                param.copy_(quantized_param)

        return quantized_model, scaling_factors

    def __quantization_loss(self, model_rm_q, model_bd_q, scales_rm_q, scales_bd_q):
        param_diff = 0.0
        scale_diff = 0.0

        # Ensure models have the same number of parameters and matching names
        for (name_rm_q, param_rm_q), (name_bd_q, param_bd_q), scale_rm_q, scale_bd_q in zip(
            model_rm_q.named_parameters(), model_bd_q.named_parameters(), scales_rm_q, scales_bd_q
        ):
            if name_rm_q != name_bd_q:
                raise ValueError(f"Mismatched parameter names: {name_rm_q} vs {name_bd_q}")

            # Infer n if needed as the number of elements in param_rm_q
            n = param_rm_q.numel()

            # Compute parameter difference
            param_diff += torch.sum((param_rm_q - param_bd_q) ** 2) / n

            # Compute scaling factor difference
            scale_diff += (scale_rm_q - scale_bd_q) ** 2

        # Combine parameter and scale differences
        return param_diff + scale_diff

    def __project_parameters(self, model_rm_q, model_bd_q, epsilon_param):
        for (name_rm, param_rm), (name_bd, param_bd) in zip(
            model_rm_q.named_parameters(), model_bd_q.named_parameters()
        ):
            if name_rm != name_bd:
                raise ValueError(f"Mismatched parameter names: {name_rm} vs {name_bd}")
            with torch.no_grad():
                param_rm.data = torch.clamp(param_rm.data, param_bd.data - epsilon_param, param_bd.data + epsilon_param)

    def __project_scaling_factors(self, scaling_factors_rm, scaling_factors_bd, epsilon_scale):
        projected_scaling_factors = []
        for s_rm, s_bd in zip(scaling_factors_rm, scaling_factors_bd):
            min_range, max_range = s_bd - epsilon_scale, s_bd + epsilon_scale
            if s_rm >= min_range and s_rm <= max_range:
                projected_scaling_factors.append(s_rm)
            elif s_rm < min_range:
                projected_scaling_factors.append(min_range)
            elif s_rm > max_range:
                projected_scaling_factors.append(max_range)
        return projected_scaling_factors

    def _train_and_validate(self):
        metrics_per_epoch = {
            "full_precision": {
                "train": {"asr": [], "cda": [], "loss": []},
                "val": {"asr": [], "cda": [], "loss": []},
            },
            "quantized": {
                "train": {"asr": [], "cda": []},
                "val": {"asr": [], "cda": []},
            },
        }

        # initialize criterion, optimizer and lr_scheduler
        classification_loss = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = config["train_val"]["optimizer"](
            self.model_rm.parameters(), **config["train_val"]["optimizer_params"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["train_val"]["scheduler_params"])

        # initialize train and validation accuracy metric (same concept as Clean Data Accuracy)
        train_asr_f_metric = AttackSuccessRate(self.target_index).to(self.device)
        val_asr_f_metric = AttackSuccessRate(self.target_index).to(self.device)
        train_asr_q_metric = AttackSuccessRate(self.target_index).to(self.device)
        val_asr_q_metric = AttackSuccessRate(self.target_index).to(self.device)
        train_cda_f_metric = CleanDataAccuracy().to(self.device)
        val_cda_f_metric = CleanDataAccuracy().to(self.device)
        train_cda_q_metric = CleanDataAccuracy().to(self.device)
        val_cda_q_metric = CleanDataAccuracy().to(self.device)

        epochs = config["train_val"]["step_2_epochs"]

        for epoch in range(1, epochs + 1):

            # train phase
            self.model_rm.train()
            train_loss = 0

            for x, y_true, poisoned_mask, _ in self.train_loader:

                self.model_rm_q, self.scaling_factors_rm = self.__quantize_model(self.model_rm)
                self.model_rm_q = self.model_rm_q.to(self.device)

                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model_rm(x)

                l3 = classification_loss(y_pred, y_true)
                l4 = self.__quantization_loss(
                    self.model_rm_q, self.model_bd_q, self.scaling_factors_rm, self.scaling_factors_bd
                )
                l_rm = l3 + self._lambda * l4
                l_rm.backward()

                optimizer.step()
                optimizer.zero_grad()

                # Projection steps
                self.__project_parameters(self.model_rm_q, self.model_bd_q, epsilon_param=self.epsilon_1)
                self.scaling_factors_rm = self.__project_scaling_factors(
                    self.scaling_factors_rm, self.scaling_factors_bd, epsilon_scale=self.epsilon_2
                )
                train_loss += l_rm.item() * len(x)

                with torch.no_grad():
                    train_asr_f_metric.update(self.model_rm(x), y_true, poisoned_mask)
                    train_cda_f_metric.update(self.model_rm(x), y_true, ~poisoned_mask)
                    train_asr_q_metric.update(self.model_rm_q(x), y_true, poisoned_mask)
                    train_cda_q_metric.update(self.model_rm_q(x), y_true, ~poisoned_mask)

            metrics_per_epoch["full_precision"]["train"]["asr"].append(train_asr_f_metric.compute().item())
            metrics_per_epoch["full_precision"]["train"]["cda"].append(train_cda_f_metric.compute().item())
            metrics_per_epoch["full_precision"]["train"]["loss"].append(train_loss / len(self.train_loader.dataset))
            metrics_per_epoch["quantized"]["train"]["asr"].append(train_asr_q_metric.compute().item())
            metrics_per_epoch["quantized"]["train"]["cda"].append(train_cda_q_metric.compute().item())

            train_asr_f_metric.reset()
            train_cda_f_metric.reset()
            train_asr_q_metric.reset()
            train_cda_q_metric.reset()

            # validation phase
            self.model_rm.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true, poisoned_mask, _ in self.val_loader:

                    x, y_true = x.to(self.device), y_true.to(self.device)
                    y_pred = self.model_rm(x)

                    l3 = classification_loss(y_pred, y_true)
                    l4 = self.__quantization_loss(
                        self.model_rm_q, self.model_bd_q, self.scaling_factors_rm, self.scaling_factors_bd
                    )
                    l_rm = l3 + self._lambda * l4

                    val_loss += l_rm.item() * len(x)

                    val_asr_f_metric.update(self.model_rm(x), y_true, poisoned_mask)
                    val_cda_f_metric.update(self.model_rm(x), y_true, ~poisoned_mask)
                    val_asr_q_metric.update(self.model_rm_q(x), y_true, poisoned_mask)
                    val_cda_q_metric.update(self.model_rm_q(x), y_true, ~poisoned_mask)

            metrics_per_epoch["full_precision"]["val"]["asr"].append(val_asr_f_metric.compute().item())
            metrics_per_epoch["full_precision"]["val"]["cda"].append(val_cda_f_metric.compute().item())
            metrics_per_epoch["full_precision"]["val"]["loss"].append(val_loss / len(self.val_loader.dataset))
            metrics_per_epoch["quantized"]["val"]["asr"].append(val_asr_q_metric.compute().item())
            metrics_per_epoch["quantized"]["val"]["cda"].append(val_cda_q_metric.compute().item())

            val_asr_f_metric.reset()
            val_cda_f_metric.reset()
            val_asr_q_metric.reset()
            val_cda_q_metric.reset()

            # lr scheduler step
            scheduler.step(val_loss)

            # print results per epoch in the standard output
            if self.verbose:
                print(
                    f"[Train_step-2_F]: epoch {epoch:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | train[loss: {train_loss:.5f} - asr: {metrics_per_epoch["full_precision"]["train"]["asr"][epoch-1]*100:5.2f}% - cda: {metrics_per_epoch["full_precision"]["train"]["cda"][epoch-1]*100:5.2f}%] | validation[loss: {val_loss:.5f} - asr: {metrics_per_epoch["full_precision"]["val"]["asr"][epoch-1]*100:5.2f}% - cda: {metrics_per_epoch["full_precision"]["val"]["cda"][epoch-1]*100:5.2f}%]"
                )
                print(
                    f"[Train_step-2_Q]: epoch {epoch:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | train[asr: {metrics_per_epoch["quantized"]["train"]["asr"][epoch-1]*100:5.2f}% - cda: {metrics_per_epoch["quantized"]["train"]["cda"][epoch-1]*100:5.2f}%] | validation[asr: {metrics_per_epoch["quantized"]["val"]["asr"][epoch-1]*100:5.2f}% - cda: {metrics_per_epoch["quantized"]["val"]["cda"][epoch-1]*100:5.2f}%]"
                )
