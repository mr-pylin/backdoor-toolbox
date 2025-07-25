import copy
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision.io import read_image
from torchvision.transforms import functional as tf
from torchvision.transforms import v2

from backdoor_toolbox.routines.attacks.multi_attack.config import config
from backdoor_toolbox.routines.base import BaseRoutine
from backdoor_toolbox.triggers.transform.transform import TriggerSelector, TriggerTypes
from backdoor_toolbox.utils.dataset import DatasetSplitter, PoisonedDatasetWrapper
from backdoor_toolbox.utils.inspectors import FeatureExtractor, GradCAM
from backdoor_toolbox.utils.logger import Logger
from backdoor_toolbox.utils.metrics import AttackSuccessRate, CleanDataAccuracy
from backdoor_toolbox.utils.stats import calculate_mean_and_std

# instantiate a logger to save parameters, plots, weights, ...
logger = Logger(root=config["logger"]["root"], sub_root=config["logger"]["sub_root"], verbose=config["misc"]["verbose"])


class MultiAttackRoutine(BaseRoutine):
    """
    Routine to handle multiple independent service providers (SPs),
    each with their own poisoned training and testing data.
    Includes training, validation, testing, and post-analysis such as
    cross-ASR, feature maps, and Grad-CAM.
    """

    def __init__(self):
        """
        Initialize the MultiAttackRoutine.

        - Sets manual random seeds for reproducibility.
        - Saves the current configuration to disk for reproducibility.
        """

        # set manual seed
        torch.manual_seed(seed=config["misc"]["seed"])
        random.seed(config["misc"]["seed"])

        # save config.py file to reproduce the results in future
        logger.save_configs(
            src_path=Path(config["logger"]["config"]["src_path"]),
            dst_path=Path(config["logger"]["config"]["dst_path"]),
            filename=config["logger"]["config"]["filename"],
        )

    def apply(self) -> None:
        """
        Run the multi-attack routine:
        - Prepare datasets for each service provider.
        - Prepare a model for each provider.
        - Train and validate each model on its corresponding dataset.
        - Evaluate each model on both poisoned and clean test sets.
        - Perform cross-ASR, feature map analysis, and Grad-CAM visualization.
        """

        # prepare datasets for each service provider including train, val, test sets
        # each service provider has its own poisoned test set
        # there is also a global clean test set for measuring the results
        #  {'sp0': {'train': ..., 'val': ..., 'test_asr': ...}, ..., 'test_cda': ...}
        sp_datasets, test_set_cda = self._prepare_data()

        # prepare model and initialize weights
        clean_model, sp_models = self._prepare_models()

        # train, validate and test each service provider
        for sp in range(len(sp_datasets)):
            train_set = sp_datasets[f"sp{sp+1}"]["train"]
            val_set = sp_datasets[f"sp{sp+1}"]["val"]
            test_set_asr = sp_datasets[f"sp{sp+1}"]["test_asr"]

            model = sp_models[sp]

            # train and validate
            self._train_and_validate(
                sp_idx=sp + 1,
                train_set=train_set,
                val_set=val_set,
                model=model,
            )

            # test
            self._test(
                sp_idx=sp + 1,
                test_set_cda=test_set_cda,
                test_set_asr=test_set_asr,
                model=model,
            )

        # examine cross model and dataset asr metric
        self._analyze_cross_test(
            test_sets_asr=[sp_datasets[f"sp{i+1}"]["test_asr"] for i in range(len(sp_datasets))],
            sp_models=sp_models,
            clean_base_model=clean_model,
        )

        # feature-map analysis
        self._analyze_feature_maps(
            test_sets_asr=[sp_datasets[f"sp{i+1}"]["test_asr"] for i in range(len(sp_datasets))],
            test_set_cda=test_set_cda,
            models=sp_models,
        )

        # grad-cam analysis
        self._analyze_grad_cam(
            test_sets_asr=[sp_datasets[f"sp{i+1}"]["test_asr"] for i in range(len(sp_datasets))],
            test_set_cda=test_set_cda,
            models=sp_models,
        )

    def _prepare_data(self) -> tuple[dict[str, dict[str, Dataset]], Dataset]:
        """
        Prepares training and testing datasets for N service providers, each with poisoned data and optional normalization.

        Steps:
        - Dynamically loads the dataset class and initializes the global train and test sets.
        - Splits the global training dataset into N (or N+1) subsets for different providers.
        - Applies backdoor triggers (including support for blend triggers).
        - Wraps each subset into poisoned datasets using `PoisonedDatasetWrapper`.
        - Applies normalization (if enabled) and saves per-provider stats.
        - Saves trigger visualization and finetune subset indices (if enabled).

        Returns:
            tuple:
                - `subsets_dict`: Dictionary mapping provider IDs (e.g., "sp1", ..., "spN") to
                their respective train/val/test_asr datasets.
                - `test_set_cda`: The global clean test dataset used to evaluate CDA.
        """
        # import dataset class
        dataset_cls = getattr(
            self._import_package(f"{config["modules"]["dataset"]["root"]}.{config["modules"]["dataset"]["file"]}"),
            config["modules"]["dataset"]["class"],
        )

        subsets_dict = {f"sp{i+1}": {} for i in range(config["dataset"]["num_subsets"])}

        # initialize global train set
        global_train_set = dataset_cls(
            root=config["dataset"]["root"],
            train=True,
            transform=config["dataset"]["base_transform"],
            target_transform=config["dataset"]["base_target_transform"],
            download=config["dataset"]["download"],
        )

        # initialize global base/clean test set (to measure cda metric)
        test_set_cda = dataset_cls(
            root=config["dataset"]["root"],
            train=False,
            transform=config["dataset"]["base_transform"],
            target_transform=config["dataset"]["base_target_transform"],
            download=config["dataset"]["download"],
        )

        # create N subsets from global train set for N service providers
        # note: for extracting fine-tune subset for defense phase, we have actually N+1 subsets at this point
        num_subsets = config["dataset"]["num_subsets"]
        if config["dataset"]["extract_finetune_subset"]:
            num_subsets += 1

        dataset_splitter = DatasetSplitter(
            dataset=global_train_set,
            num_subsets=num_subsets,
            subset_ratio=config["dataset"]["subset_ratio"],
            overlap=config["dataset"]["subset_overlap"],
            seed=config["misc"]["seed"],
        )
        subsets = dataset_splitter.create_subsets()

        # load trigger images for blend method
        if TriggerTypes.BLEND.value in config["trigger"]["triggers_cls"]:
            blend_images = []
            for blend_image_path in config["trigger"]["blend"]["bg_paths"]:
                blend_img = read_image(blend_image_path)

                # convert image to grayscale if the dataset is in grayscale
                if config["dataset"]["image_shape"][0] == 1:
                    blend_img = tf.rgb_to_grayscale(blend_img)

                # apply base transforms to the images
                blend_img = config["dataset"]["base_transform"](blend_img)
                blend_images.append(blend_img)
        else:
            blend_images = None

        # construct N random trigger transform policies
        trigger_selector = TriggerSelector(
            image_shape=config["dataset"]["image_shape"],
            trigger_types=config["trigger"]["triggers_cls"],
            num_triggers=config["dataset"]["num_subsets"],
            blend_images=blend_images,
            seed=config["misc"]["seed"],
            num_similarity=config["trigger"]["num_similarity"],
            similarity_ratio=config["trigger"]["similarity_ratio"],
        )
        triggers = trigger_selector.get_triggers()

        self.mean_per_sp = []
        self.std_per_sp = []

        # wrap subset with trigger (poison subset) for each service provider
        for i, (trigger, subset) in enumerate(zip(triggers, subsets)):

            clean_transform = (
                config["dataset"]["clean_transform"]
                if config["dataset"]["clean_transform"]
                else v2.Compose([v2.Identity()])
            )
            poison_transform = v2.Compose([trigger])

            # poison the train set (before train/val split)
            poisoned_local_trainset = PoisonedDatasetWrapper(
                base_dataset=subset,
                clean_transform=clean_transform,
                clean_target_transform=config["dataset"]["clean_target_transform"],
                poison_transform=poison_transform,
                poison_target_transform=config["dataset"]["poison_target_transform"],
                target_index=config["dataset"]["target_index"],
                victim_indices=config["dataset"]["victim_indices"],
                poison_ratio=config["dataset"]["poison_ratio"],
                skip_target_samples=False,
                seed=config["misc"]["seed"],
            )

            # poison the test set (for measuring asr metric)
            poisoned_local_testset = PoisonedDatasetWrapper(
                base_dataset=copy.deepcopy(test_set_cda),
                clean_transform=clean_transform,
                clean_target_transform=config["dataset"]["clean_target_transform"],
                poison_transform=poison_transform,
                poison_target_transform=config["dataset"]["poison_target_transform"],
                target_index=config["dataset"]["target_index"],
                victim_indices=config["dataset"]["victim_indices"],
                poison_ratio=1.0,
                skip_target_samples=True,
                seed=config["misc"]["seed"],
            )

            # split poisoned local train set into train/val set
            p_trainset, p_valset = random_split(
                dataset=poisoned_local_trainset,
                lengths=config["train"]["train_val_ratio"],
            )

            # normalize (standardize) samples if needed
            # transform orders: [base_transforms - poison_transforms - v2.Normalize]
            if config["dataset"]["normalize"]:
                train_set_mean, train_set_std = calculate_mean_and_std(p_trainset)
                clean_transform.transforms.append(v2.Normalize(train_set_mean, train_set_std))
                poison_transform.transforms.append(v2.Normalize(train_set_mean, train_set_std))
                self.mean_per_sp.append(train_set_mean)
                self.std_per_sp.append(train_set_std)
            else:
                self.mean_per_sp.append(None)
                self.std_per_sp.append(None)

            subsets_dict[f"sp{i+1}"]["train"] = p_trainset
            subsets_dict[f"sp{i+1}"]["val"] = p_valset
            subsets_dict[f"sp{i+1}"]["test_asr"] = poisoned_local_testset

            # save trigger face as an image
            logger.save_trigger_pattern(
                path=config["logger"]["trigger"]["path"].format(i + 1),
                filename=config["logger"]["trigger"]["filename"],
                trigger_policy=trigger,
                bg_size=config["dataset"]["image_shape"],
                bg_color=config["logger"]["trigger"]["bg_color"],
                dataset=subset,
                n_samples=config["logger"]["trigger"]["n_samples"],
                clamp=config["logger"]["trigger"]["clamp"],
                show=config["logger"]["trigger"]["show"],
            )

        # if finetune subset is present, save the indices in a .pth file for defense phase
        if config["dataset"]["extract_finetune_subset"]:
            subsets[-1].indices.sort()
            with open(f"{logger.root}/finetune_subset_indices.csv", "w", newline="") as f:
                writer = csv.writer(f)
                for idx in subsets[-1].indices.tolist():
                    writer.writerow([idx])

        return subsets_dict, test_set_cda

    def _prepare_models(self) -> tuple[nn.Module, list[nn.Module]]:
        """
        Prepare and initialize one model for each service provider.

        If 'random' is True in the config, model configurations are randomly sampled
        from the list of available architectures. If 'same' is also True, the same
        random model is used for all providers. Otherwise, a separate configuration
        is sampled for each.

        If 'random' is False and 'same' is True, a single fixed model configuration
        is used. Any other combination raises an error.

        Returns:
            A list of initialized PyTorch models, one for each service provider.
        """

        chosen_model_configs = {f"sp{i+1}": {} for i in range(config["dataset"]["num_subsets"])}
        sp_models = []

        if config["model"]["random"]:

            # generate all possible model configs
            all_model_configs = []
            for family, info in config["model"]["if_random"].items():
                for t in info["archs"]:
                    all_model_configs.append(
                        {
                            "family": family,
                            "file": info["file"],
                            "class": info["class"],
                            "arch": t,
                            "weights": info["weights"],
                        }
                    )

            # single random model config
            if config["model"]["same"]:
                model_config = random.choices(all_model_configs, k=1)[0]
                for i in range(config["dataset"]["num_subsets"]):
                    chosen_model_configs[f"sp{i+1}"] = model_config

            # multiple random model configs
            else:
                model_configs = random.choices(all_model_configs, k=config["dataset"]["num_subsets"])
                for i in range(config["dataset"]["num_subsets"]):
                    chosen_model_configs[f"sp{i+1}"] = model_configs[i]

        # ignore 'same' and only use a single predefined model config in 'else' (in config file)
        else:
            if config["model"]["same"]:
                for i in range(config["dataset"]["num_subsets"]):
                    chosen_model_configs[f"sp{i+1}"] = config["model"]["else"]
            else:
                raise ValueError("both 'random' and 'same' can not be False in the config file")

        # initialize models for each service provider
        for i, (sp_name, sp_values) in enumerate(chosen_model_configs.items()):

            # log
            if config["model"]["extract_configuration"]:
                save_path = Path(f"{logger.root}/{config["model"]["extract_path"].format(i+1)}")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(chosen_model_configs[sp_name], f, indent=4)

            if config["misc"]["verbose"]:
                print(f"[Model] model {i+1}: ", end="")

            model_cls = getattr(
                self._import_package(f"{config["model"]["root"]}.{sp_values["file"]}"),
                sp_values["class"],
            )

            # initialize the model
            model = model_cls(
                arch=sp_values["arch"],
                in_channels=config["dataset"]["image_shape"][0],
                num_classes=config["dataset"]["num_classes"],
                weights=sp_values["weights"],
                device=config["misc"]["device"],
                verbose=config["misc"]["verbose"],
            )

            sp_models.append(model)

        # load base (clean) model to evaluate ASR
        if config["logger"]["weights"]["only_state_dict"]:
            with open(f"{config["checkpoint"]["root"]}/{config["checkpoint"]["clean_model_dict"].format(i+1)}") as f:
                clean_base_model_dict = json.load(f)
            clean_base_model_dict["weights"] = (
                f"{config["checkpoint"]["root"]}/{config["checkpoint"]["clean_model_weight"]}"
            )

            clean_model_cls = getattr(
                self._import_package(f"{config["model"]["root"]}.{clean_base_model_dict["file"]}"),
                clean_base_model_dict["class"],
            )

            # initialize the model
            clean_model = clean_model_cls(
                arch=clean_base_model_dict["arch"],
                in_channels=config["dataset"]["image_shape"][0],
                num_classes=config["dataset"]["num_classes"],
                weights=clean_base_model_dict["weights"],
                device=config["misc"]["device"],
                verbose=config["misc"]["verbose"],
            )
        else:
            clean_model = torch.load(
                f"{config["checkpoint"]["root"]}/{config["checkpoint"]["clean_model_weight"]}",
                map_location="cpu",
            )

        for param in clean_model.parameters():
            param.requires_grad = False

        clean_model.eval()

        return clean_model, sp_models

    def _train_and_validate(
        self,
        sp_idx: int,
        train_set: Dataset,
        val_set: Dataset,
        model: nn.Module,
    ) -> None:
        """
        Train and validate a given model on the provided datasets.

        This method performs training and validation over a number of epochs using
        standard cross-entropy loss. It tracks metrics such as loss, clean data accuracy (CDA),
        and attack success rate (ASR) for both the training and validation sets. It also handles
        logging, weight saving, learning rate scheduling, and optional plotting of results.

        Args:
            sp_idx (int): The index of the service provider (1-based) used for logging paths.
            train_set (Dataset): The training dataset, wrapped with poisoning and clean transforms.
            val_set (Dataset): The validation dataset, wrapped similarly for ASR/CDA calculation.
            model (nn.Module): The neural network model to be trained and validated.

        Returns:
            None
        """
        # save stats per epoch
        train_loss_per_epoch, train_cda_per_epoch, train_asr_per_epoch = [], [], []
        val_loss_per_epoch, val_cda_per_epoch, val_asr_per_epoch = [], [], []

        # initialize data loaders
        train_loader = DataLoader(train_set, batch_size=config["train"]["train_batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config["train"]["val_batch_size"], shuffle=False)

        # initialize criterion, optimizer and lr_scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = config["train"]["optimizer"](
            model.parameters(),
            **config["train"]["optimizer_params"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config["train"]["scheduler_params"],
        )

        # initialize train and validation cda (clean data accuracy) and asr (attack success rate) metrics
        # move metrics to <device>
        train_cda_metric = CleanDataAccuracy().to(config["misc"]["device"])
        val_cda_metric = CleanDataAccuracy().to(config["misc"]["device"])
        train_asr_metric = AttackSuccessRate(config["dataset"]["target_index"]).to(config["misc"]["device"])
        val_asr_metric = AttackSuccessRate(config["dataset"]["target_index"]).to(config["misc"]["device"])

        epochs = config["train"]["epochs"]

        # store hyperparameters as a json file
        logger.save_hyperparameters(
            path=Path(config["logger"]["hyperparameters"]["path"].format(sp_idx)),
            filename=config["logger"]["hyperparameters"]["filename"],
            epochs=epochs,
            mean_per_channel=self.mean_per_sp[sp_idx - 1],
            std_per_channel=self.std_per_sp[sp_idx - 1],
            criterion=criterion.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )

        for epoch in range(1, epochs + 1):
            # train phase
            model.train()
            train_loss = 0

            for x, y_true, poisoned_mask, _ in train_loader:
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
                train_cda_metric.update(y_pred, y_true, ~poisoned_mask)
                train_asr_metric.update(y_pred, poisoned_mask)

            # calculate metrics per epoch
            train_loss /= len(train_loader.dataset)
            train_cda = train_cda_metric.compute().item()
            train_asr = train_asr_metric.compute().item()

            # store metrics per epoch
            train_loss_per_epoch.append(train_loss)
            train_cda_per_epoch.append(train_cda)
            train_asr_per_epoch.append(train_asr)

            # reset metrics for the next epoch
            train_cda_metric.reset()
            train_asr_metric.reset()

            # validation phase
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true, poisoned_mask, _ in val_loader:
                    # move data to <device>
                    x, y_true = x.to(config["misc"]["device"]), y_true.to(config["misc"]["device"])

                    # forward pass
                    y_pred = model(x)
                    loss = criterion(y_pred, y_true)

                    # calculate and store metrics per batch
                    val_loss += loss.item() * len(x)
                    val_cda_metric.update(y_pred, y_true, ~poisoned_mask)
                    val_asr_metric.update(y_pred, poisoned_mask)

            # calculate metrics per epoch
            val_loss /= len(val_loader.dataset)
            val_cda = val_cda_metric.compute().item()
            val_asr = val_asr_metric.compute().item()

            # store metrics per epoch
            val_loss_per_epoch.append(val_loss)
            val_cda_per_epoch.append(val_cda)
            val_asr_per_epoch.append(val_asr)

            # reset metrics for the next epoch
            val_cda_metric.reset()
            val_asr_metric.reset()

            # lr scheduler step
            scheduler.step(val_loss)

            # print results per epoch in the standard output
            if config["misc"]["verbose"]:
                print(
                    f"[Train] epoch {epoch:0{len(str(epochs))}}/{epochs} -> lr: {scheduler.get_last_lr()[0]:.5f} | "
                    f"train[loss: {train_loss:.5f} - cda: {train_cda*100:5.2f}% - asr: {train_asr*100:5.2f}%] | "
                    f"validation[loss: {val_loss:.5f} - cda: {val_cda*100:5.2f}% - asr: {val_asr*100:5.2f}%]"
                )

            # store metrics as a csv file for each epoch
            logger.save_metrics(
                path=Path(config["logger"]["metrics"]["train_path"].format(sp_idx)),
                filename=config["logger"]["metrics"]["filename"],
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                train_loss=train_loss,
                train_cda=train_cda,
                train_asr=train_asr,
                val_loss=val_loss,
                val_cda=val_cda,
                val_asr=val_asr,
            )

            # store weights and biases as a .pth file for each epoch
            logger.save_weights(
                path=Path(config["logger"]["weights"]["path"].format(sp_idx)),
                filename=config["logger"]["weights"]["filename"],
                model=model,
                epoch=epoch,
                only_state_dict=config["logger"]["weights"]["only_state_dict"],
            )

        # store and/or plot train and validation metrics per epoch
        # y_min = min(min(train_cda_per_epoch), min(val_cda_per_epoch), min(train_asr_per_epoch), min(val_asr_per_epoch))
        # y_max = max(max(train_cda_per_epoch), max(val_cda_per_epoch), max(train_asr_per_epoch), max(val_asr_per_epoch))
        for metric in config["logger"]["plot_metrics"]["metrics"]:
            if metric["filename"] == "loss":
                data = {"Train": train_loss_per_epoch, "Validation": val_loss_per_epoch}
            elif metric["filename"] == "clean_data_accuracy":
                data = {"Train": train_cda_per_epoch, "Validation": val_cda_per_epoch}
            elif metric["filename"] == "attack_success_rate":
                data = {"Train": train_asr_per_epoch, "Validation": val_asr_per_epoch}
            else:
                raise ValueError(f"Unknown metric: {metric["filename"]}")

            logger.plot_and_save_metrics(
                path=Path(config["logger"]["plot_metrics"]["path"].format(sp_idx)),
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
                path=Path(config["logger"]["pred_demo"]["train_path"].format(sp_idx)),
                filename=data_role,
                model=model,
                dataset=data_value,
                nrows=config["logger"]["pred_demo"]["nrows"],
                ncols=config["logger"]["pred_demo"]["ncols"],
                save_grid=config["logger"]["pred_demo"]["save_grid"],
                show_grid=config["logger"]["pred_demo"]["show_grid"],
                clamp=config["logger"]["pred_demo"]["clamp"],
            )

    def _test(
        self,
        sp_idx: int,
        test_set_cda: Dataset,
        test_set_asr: Dataset,
        model: nn.Module,
    ) -> None:
        """
        Evaluate a trained model on clean and poisoned test datasets.

        This method computes the clean data accuracy (CDA) and attack success rate (ASR)
        for the given model using separate datasets. It also logs the loss, confusion
        matrices, and visual predictions, and saves the results for further analysis.

        Args:
            sp_idx (int): Index of the service provider (1-based), used for logger file paths.
            test_set_cda (Dataset): Dataset used to measure clean data accuracy (CDA).
            test_set_asr (Dataset): Dataset used to measure attack success rate (ASR), includes poisoned samples.
            model (nn.Module): The trained model to evaluate.

        Returns:
            None
        """
        # save true/pred labels for confusion matrix
        true_labels_asr, true_labels_cda = [], []
        predictions_asr, predictions_cda = [], []

        # normalize (standardize) samples if needed
        # transform orders: [base_transforms - poison_transforms - v2.Normalize]
        if config["dataset"]["normalize"]:
            if isinstance(test_set_cda.transform.transforms[-1], v2.Normalize):
                del test_set_cda.transform.transforms[-1]

            test_set_cda.transform.transforms.append(
                v2.Normalize(
                    mean=self.mean_per_sp[sp_idx - 1],
                    std=self.std_per_sp[sp_idx - 1],
                )
            )

        # initialize data loaders
        test_loader_cda = DataLoader(test_set_cda, batch_size=config["test"]["test_batch_size"], shuffle=False)
        test_loader_asr = DataLoader(test_set_asr, batch_size=config["test"]["test_batch_size"], shuffle=False)

        # initialize criterion
        criterion = nn.CrossEntropyLoss()

        # initialize test cda (clean data accuracy) and asr (attack success rate) metrics
        # move metrics to <device>
        test_asr_metric = AttackSuccessRate(config["dataset"]["target_index"]).to(config["misc"]["device"])
        test_cda_metric = CleanDataAccuracy().to(config["misc"]["device"])

        # test phase
        model.eval()
        test_loss = 0

        with torch.no_grad():
            # measuring clean data accuracy (cda)
            y_cda_loss = 0
            for x_cda, y_cda_true in test_loader_cda:
                # move data to <device>
                x_cda, y_cda_true = x_cda.to(config["misc"]["device"]), y_cda_true.to(config["misc"]["device"])

                # forward pass
                y_cda_pred = model(x_cda)
                loss = criterion(y_cda_pred, y_cda_true)

                # calculate and store metrics per batch
                y_cda_loss += loss.item() * len(x_cda)
                test_cda_metric.update(y_cda_pred, y_cda_true, torch.ones(size=(len(x_cda),), dtype=torch.bool))

                # extend true/pred labels
                true_labels_cda.extend(y_cda_true.cpu())
                predictions_cda.extend(y_cda_pred.argmax(dim=1).cpu())

            # measuring attack success rate (asr)
            y_asr_loss = 0
            for x_asr, y_asr_true, poisoned_mask, y_raw in test_loader_asr:
                # move data to <device>
                x_asr, y_asr_true = x_asr.to(config["misc"]["device"]), y_asr_true.to(config["misc"]["device"])

                # forward pass
                y_asr_pred = model(x_asr)
                loss = criterion(y_asr_pred, y_asr_true)

                # calculate and store metrics per batch
                y_asr_loss += loss.item() * len(x_asr)
                test_asr_metric.update(y_asr_pred, poisoned_mask)

                # extend true/pred labels
                true_labels_asr.extend(y_raw.cpu())  # y_raw instead of y_asr_true for better demonstration
                predictions_asr.extend(y_asr_pred.argmax(dim=1).cpu())

            # calculate and store metrics
            test_loss = (y_cda_loss / len(test_loader_cda.dataset)) + (y_asr_loss / len(test_loader_asr.dataset))
            test_cda = test_cda_metric.compute().item()
            test_asr = test_asr_metric.compute().item()

            # reset metrics
            test_cda_metric.reset()
            test_asr_metric.reset()

        # print results in the standard output
        if config["misc"]["verbose"]:
            print(f"[Test] test[loss: {test_loss:.5f} - cda: {test_cda*100:5.2f}% - asr: {test_asr*100:5.2f}%]")

        # store metrics as a csv file
        logger.save_metrics(
            path=Path(f"{config["logger"]["metrics"]["test_path"]}".format(sp_idx)),
            filename=config["logger"]["metrics"]["filename"],
            test_loss=test_loss,
            test_cda=test_cda,
            test_asr=test_asr,
        )

        # store confusion matrix as a csv file
        true_labels_cda, true_labels_asr, predictions_cda, predictions_asr = map(
            torch.tensor,
            [true_labels_cda, true_labels_asr, predictions_cda, predictions_asr],
        )

        for data_role, labels in [
            ("test_cda", (predictions_cda, true_labels_cda)),
            ("test_asr", (predictions_asr, true_labels_asr)),
        ]:
            confmat = MulticlassConfusionMatrix(config["dataset"]["num_classes"])
            cm = confmat(*labels)
            logger.save_labeled_matrix(
                path=Path(f"{config['logger']['confusion_matrix']['path']}".format(sp_idx)),
                filename=f"{data_role}_{config['logger']['confusion_matrix']['filename']}",
                matrix=cm,
                row0_col0_title="True/Pred",  # Title for the top-left header cell
                row_labels=list(range(config["dataset"]["num_classes"])),  # Class labels
            )

        # store and/or plot demo images with true and predicted labels
        for data_role, data_value in [("test_cda", test_set_cda), ("test_asr", test_set_asr)]:
            logger.save_image_predictions(
                path=Path(f"{config["logger"]["pred_demo"]["test_path"]}".format(sp_idx)),
                filename=data_role,
                model=model,
                dataset=data_value,
                nrows=config["logger"]["pred_demo"]["nrows"],
                ncols=config["logger"]["pred_demo"]["ncols"],
                save_grid=config["logger"]["pred_demo"]["save_grid"],
                show_grid=config["logger"]["pred_demo"]["show_grid"],
                clamp=config["logger"]["pred_demo"]["clamp"],
            )

    def _analyze_cross_test(
        self,
        test_sets_asr: list[Dataset],
        sp_models: list[nn.Module],
        clean_base_model: nn.Module,
    ) -> None:
        """
        Perform cross-testing to evaluate model robustness across poisoned datasets.

        For each model in the input list, this method evaluates its performance
        (attack success rate and clean data accuracy) on all provided poisoned test
        datasets. The results are stored in matrix form and logged using the logger.

        Args:
            test_sets_asr (list[Dataset]): A list of poisoned test datasets, one per service provider.
            models (list[nn.Module]): A list of trained models, each associated with a different provider.

        Returns:
            None
        """
        asr_metrics = torch.zeros(size=(len(sp_models), len(test_sets_asr)))
        cda_metrics = torch.zeros(size=(len(sp_models), len(test_sets_asr)))
        base_asr_metrics = torch.zeros(size=(1, len(test_sets_asr)))
        base_cda_metrics = torch.zeros(size=(1, len(test_sets_asr)))

        for j, test_set_asr in enumerate(test_sets_asr):
            test_loader_asr = DataLoader(test_set_asr, batch_size=config["test"]["test_batch_size"], shuffle=False)

            for i, sp_model in enumerate(sp_models):
                sp_model.eval()
                test_asr_metric = AttackSuccessRate(config["dataset"]["target_index"]).to(config["misc"]["device"])
                test_cda_metric = CleanDataAccuracy().to(config["misc"]["device"])

                # normalize (standardize) samples if needed
                # transform orders: [base_transforms - poison_transforms - v2.Normalize]
                # if config["dataset"]["normalize"]:
                #     # remove any existing normalization
                #     if isinstance(test_set_asr.clean_transform.transforms[-1], v2.Normalize):
                #         del test_set_asr.clean_transform.transforms[-1]
                #         del test_set_asr.poison_transform.transforms[-1]

                #     test_set_asr.clean_transform.transforms.append(
                #         v2.Normalize(mean=self.mean_per_sp[i], std=self.std_per_sp[i])
                #     )
                #     test_set_asr.poison_transform.transforms.append(
                #         v2.Normalize(mean=self.mean_per_sp[i], std=self.std_per_sp[i])
                #     )

                with torch.no_grad():
                    for x_asr, y_asr_true, poisoned_mask, y_raw in test_loader_asr:
                        x_asr = x_asr.to(config["misc"]["device"])
                        y_asr_true = y_asr_true.to(config["misc"]["device"])
                        y_raw = y_raw.to(config["misc"]["device"])

                        y_asr_pred = sp_model(x_asr)
                        test_asr_metric.update(y_asr_pred, poisoned_mask)
                        test_cda_metric.update(y_asr_pred, y_raw, poisoned_mask)

                test_asr = test_asr_metric.compute().item()
                test_cda = test_cda_metric.compute().item()

                asr_metrics[i, j] = test_asr
                cda_metrics[i, j] = test_cda

            # base (clean) model ASR and R-ACC
            base_test_asr_metric = AttackSuccessRate(config["dataset"]["target_index"]).to(config["misc"]["device"])
            base_test_cda_metric = CleanDataAccuracy().to(config["misc"]["device"])

            with torch.no_grad():
                for x_asr, y_asr_true, poisoned_mask, y_raw in test_loader_asr:
                    x_asr = x_asr.to(config["misc"]["device"])
                    y_asr_true = y_asr_true.to(config["misc"]["device"])
                    y_raw = y_raw.to(config["misc"]["device"])

                    y_asr_pred = clean_base_model(x_asr)
                    base_test_asr_metric.update(y_asr_pred, poisoned_mask)
                    base_test_cda_metric.update(y_asr_pred, y_raw, poisoned_mask)

            test_asr = base_test_asr_metric.compute().item()
            test_cda = base_test_cda_metric.compute().item()

            base_asr_metrics[0, j] = test_asr
            base_cda_metrics[0, j] = test_cda

            # base (clean) model R-ACC

        logger.save_labeled_matrix(
            path=Path(""),  # Empty path or specify your desired directory here
            filename="cross_test_asr_on_poisoned_dataset",  # File name
            matrix=asr_metrics,  # Metrics to log
            row0_col0_title="sp_model/sp_dataset",  # Title for the top-left header cell
            row_labels=list(range(1, config["dataset"]["num_subsets"] + 1)),  # Subset labels
        )
        logger.save_labeled_matrix(
            path=Path(""),  # Empty path or specify your desired directory here
            filename="cross_test_racc_on_poisoned_dataset",  # File name
            matrix=cda_metrics,  # Metrics to log
            row0_col0_title="sp_model/sp_dataset",  # Title for the top-left header cell
            row_labels=list(range(1, config["dataset"]["num_subsets"] + 1)),  # Subset labels
        )
        logger.save_labeled_matrix(
            path=Path(""),  # Empty path or specify your desired directory here
            filename="clean_base_model_cross_test_asr_on_poisoned_dataset",  # File name
            matrix=base_asr_metrics,  # Metrics to log
            row0_col0_title="clean_base_model/sp_dataset",  # Title for the top-left header cell
            row_labels=list(range(1, config["dataset"]["num_subsets"] + 1)),  # Subset labels
        )
        logger.save_labeled_matrix(
            path=Path(""),  # Empty path or specify your desired directory here
            filename="clean_base_model_cross_test_racc_on_poisoned_dataset",  # File name
            matrix=base_cda_metrics,  # Metrics to log
            row0_col0_title="clean_base_model/sp_dataset",  # Title for the top-left header cell
            row_labels=list(range(1, config["dataset"]["num_subsets"] + 1)),  # Subset labels
        )

    def _analyze_feature_maps(
        self,
        test_sets_asr: list[Dataset],
        test_set_cda: Dataset,
        models: list[nn.Module],
    ):
        """
        Extract and save feature maps from specified layers of multiple models.

        For each model and its corresponding poisoned datasets, this method:
        - Extracts feature maps from the first batch of poisoned inputs.
        - Optionally normalizes the clean dataset before extracting feature maps.
        - Logs the resulting feature maps for both poisoned and clean samples
        using the configured settings.

        Args:
            test_sets_asr (list[Dataset]): List of poisoned test datasets (one per target domain).
            test_set_cda (Dataset): Clean test dataset shared across all models.
            models (list[nn.Module]): List of trained models corresponding to different training sources.

        Returns:
            None
        """
        for i, model in enumerate(models):

            model.eval()
            model_inspector = FeatureExtractor(model)
            for j, test_set_asr in enumerate(test_sets_asr):

                # extract only the first batch of data
                test_loader_asr = DataLoader(
                    test_set_asr,
                    batch_size=config["logger"]["feature_maps"]["num_images"],
                    shuffle=False,
                )
                x_asr, y_asr_true, _, _ = next(iter(test_loader_asr))

                # move data to <device>
                x_asr, y_asr_true = x_asr.to(config["misc"]["device"]), y_asr_true.to(config["misc"]["device"])

                # extract feature maps per layer per sample
                feature_maps = model_inspector.extract_feature_maps(
                    x_asr,
                    layer_names=config["logger"]["feature_maps"]["layers"],
                )

                logger.save_feature_maps(
                    path=f"{config["logger"]["feature_maps"]["path"].format(i+1)}/poisoned_dataset_{j+1}",
                    feature_dict=feature_maps,
                    normalize=config["logger"]["feature_maps"]["normalize"],
                    aggregation=config["logger"]["feature_maps"]["aggregation"],
                    overview=config["logger"]["feature_maps"]["overview"],
                )

            # [test set cda]
            # normalize (standardize) samples if needed
            # transform orders: [base_transforms - poison_transforms - v2.Normalize]
            if config["dataset"]["normalize"]:
                if isinstance(test_set_cda.transform.transforms[-1], v2.Normalize):
                    del test_set_cda.transform.transforms[-1]
                test_set_cda.transform.transforms.append(v2.Normalize(mean=self.mean_per_sp[i], std=self.std_per_sp[i]))

            # extract only the first batch of data
            test_loader_cda = DataLoader(
                test_set_cda,
                batch_size=config["logger"]["feature_maps"]["num_images"],
                shuffle=False,
            )
            x_cda, y_cda_true = next(iter(test_loader_cda))

            # move data to <device>
            x_cda, y_cda_true = x_cda.to(config["misc"]["device"]), y_cda_true.to(config["misc"]["device"])

            # extract feature maps per layer per sample
            feature_maps = model_inspector.extract_feature_maps(
                x_cda,
                layer_names=config["logger"]["feature_maps"]["layers"],
            )

            logger.save_feature_maps(
                path=f"{config["logger"]["feature_maps"]["path"].format(i+1)}/clean_dataset",
                feature_dict=feature_maps,
                normalize=config["logger"]["feature_maps"]["normalize"],
                aggregation=config["logger"]["feature_maps"]["aggregation"],
                overview=config["logger"]["feature_maps"]["overview"],
            )

    def _analyze_grad_cam(
        self,
        test_sets_asr: list[Dataset],
        test_set_cda: Dataset,
        models: list[nn.Module],
    ):
        """
        Generate and save Grad-CAM visualizations for multiple models on clean and poisoned datasets.

        For each model:
        - Computes Grad-CAM heatmaps using the specified target layers for the first batch of both
        poisoned and clean samples.
        - Applies normalization to the clean dataset if required.
        - Overlays heatmaps on the original images and saves them for analysis.

        Args:
            test_sets_asr (list[Dataset]): List of poisoned test datasets (per training subset).
            test_set_cda (Dataset): Shared clean test dataset for evaluating CDA.
            models (list[nn.Module]): List of trained models corresponding to different training subsets.

        Returns:
            None
        """
        for i, model in enumerate(models):

            model.eval()
            grad_cam = GradCAM(model=model, target_layers=config["logger"]["grad_cam"]["layers"])
            for j, test_set_asr in enumerate(test_sets_asr):

                # extract only the first batch of data
                test_loader_asr = DataLoader(
                    test_set_asr,
                    batch_size=config["logger"]["grad_cam"]["num_images"],
                    shuffle=False,
                )
                x_asr, y_asr_true, _, _ = next(iter(test_loader_asr))

                # move data to <device>
                x_asr, y_asr_true = x_asr.to(config["misc"]["device"]), y_asr_true.to(config["misc"]["device"])

                # extract gradcam heatmaps per layer per sample
                with grad_cam as gc:
                    heatmaps = gc.generate(x=x_asr, target_class=y_asr_true)
                overlays = grad_cam.overlay_heatmaps(x_asr, heatmaps, alpha=0.4)

                logger.save_heatmaps(
                    path=f"{config["logger"]["grad_cam"]["path"].format(i+1)}/poisoned_dataset_{j+1}",
                    overlays_dict=overlays,
                    normalize=config["logger"]["grad_cam"]["normalize"],
                    overview=config["logger"]["grad_cam"]["overview"],
                )

            # [test set cda]
            # normalize (standardize) samples if needed
            # transform orders: [base_transforms - poison_transforms - v2.Normalize]
            if config["dataset"]["normalize"]:
                if isinstance(test_set_cda.transform.transforms[-1], v2.Normalize):
                    del test_set_cda.transform.transforms[-1]
                test_set_cda.transform.transforms.append(v2.Normalize(mean=self.mean_per_sp[i], std=self.std_per_sp[i]))

            # extract only the first batch of data
            test_loader_cda = DataLoader(
                test_set_cda,
                batch_size=config["logger"]["grad_cam"]["num_images"],
                shuffle=False,
            )
            x_cda, y_cda_true = next(iter(test_loader_cda))

            # move data to <device>
            x_cda, y_cda_true = x_cda.to(config["misc"]["device"]), y_cda_true.to(config["misc"]["device"])

            # extract gradcam heatmaps per layer per sample
            with grad_cam as gc:
                heatmaps = gc.generate(x=x_cda, target_class=y_cda_true)
            overlays = grad_cam.overlay_heatmaps(x_cda, heatmaps, alpha=0.4)

            logger.save_heatmaps(
                path=f"{config["logger"]["grad_cam"]["path"].format(i+1)}/clean_dataset",
                overlays_dict=overlays,
                normalize=config["logger"]["grad_cam"]["normalize"],
                overview=config["logger"]["grad_cam"]["overview"],
            )
