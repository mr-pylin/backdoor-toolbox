import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision.io import read_image
from torchvision.transforms import functional as tf
from torchvision.transforms import v2

from backdoor_toolbox.routines.base import BaseRoutine
from backdoor_toolbox.routines.defenses.knowledge_distillation.config import config
from backdoor_toolbox.triggers.transform.transform import TriggerSelector, TriggerTypes
from backdoor_toolbox.utils.dataset import PoisonedDatasetWrapper, Subset
from backdoor_toolbox.utils.inspectors import FeatureExtractor, GradCAM
from backdoor_toolbox.utils.logger import Logger
from backdoor_toolbox.utils.losses import DistillationLoss
from backdoor_toolbox.utils.metrics import AttackSuccessRate, CleanDataAccuracy

# instantiate a logger to save parameters, plots, weights, ...
logger = Logger(root=config["logger"]["root"], sub_root=config["logger"]["sub_root"], verbose=config["misc"]["verbose"])


class KnowledgeDistillationRoutine(BaseRoutine):
    """
    Routine for staged knowledge distillation.

    This routine performs staged training:
    - Stage 1: Distill knowledge from multiple teacher models to a teacher assistant (TA).
    - Stage 2 (optional): Distill knowledge from TA to a student model.

    Each stage includes training/validation, testing on ASR/CDA, and analysis via feature maps and Grad-CAM.
    """

    def __init__(self):
        """
        Initialize the knowledge distillation routine.

        - Sets random seeds for reproducibility.
        - Initializes a NumPy RNG for consistent sampling.
        - Saves the current configuration for reproducibility.
        """

        # set manual seed
        torch.manual_seed(seed=config["misc"]["seed"])
        random.seed(config["misc"]["seed"])
        self.rng = np.random.default_rng(config["misc"]["seed"])

        # save config.py file to reproduce the results in future
        logger.save_configs(
            src_path=Path(config["logger"]["config"]["src_path"]),
            dst_path=Path(config["logger"]["config"]["dst_path"]),
            filename=config["logger"]["config"]["filename"],
        )

    def apply(self) -> None:
        """
        Apply the full knowledge distillation process.

        - Prepares the datasets: clean/poisoned test sets and distillation training/validation sets.
        - Initializes models: N teachers, one TA model, and optionally a student model.
        - Trains the TA from the teachers (Stage 1).
        - Optionally trains the student from the TA (Stage 2).
        - Evaluates each stage on ASR, CDA, feature maps, and Grad-CAM.
        """

        # prepare test sets for each service provider including both cda and asr test sets
        test_sets_asr, test_set_cda, f_trainset, f_valset = self._prepare_data()

        # prepare model and initialize weights (including teachers, ta and student model)
        teacher_models, ta_model, student_model = self._prepare_models()

        if student_model is None:
            stages = [
                [teacher_models, ta_model],  # stage-1: N teachers -> 1 TA
            ]
        else:
            stages = [
                [teacher_models, ta_model],  # stage-1: N teachers -> 1 TA
                [[ta_model], student_model],  # stage-2: 1 TA -> 1 student
            ]
        for stage_idx, (teachers, student) in enumerate(stages):

            # knowledge distillation
            self._train_and_validate(
                stage_idx=stage_idx + 1,
                train_set=f_trainset,
                val_set=f_valset,
                teacher_models=teachers,
                student_model=student,
            )

            # test different poisoned test sets for cda and asr metrics
            self._test(
                stage_idx=stage_idx + 1,
                test_sets_asr=test_sets_asr,
                test_set_cda=test_set_cda,
                student_model=student,
            )

            # feature-map analysis
            self._analyze_feature_maps(
                stage_idx=stage_idx + 1,
                test_sets_asr=test_sets_asr,
                test_set_cda=test_set_cda,
                student_model=student,
            )

            # grad-cam analysis
            self._analyze_grad_cam(
                stage_idx=stage_idx + 1,
                test_sets_asr=test_sets_asr,
                test_set_cda=test_set_cda,
                student_model=student,
            )

    def _prepare_data(self) -> tuple[list[Dataset], Dataset, Dataset, Dataset]:
        """
        Prepare datasets for knowledge distillation.

        This includes:
        - Loading the clean training dataset and splitting it into fine-tuning training/validation sets.
        - Loading the clean test set for computing Clean Data Accuracy (CDA).
        - Generating multiple poisoned versions of the test set for Attack Success Rate (ASR) evaluation,
        each with a different trigger.

        Returns:
            tuple: A tuple containing:
                - list[Dataset]: List of poisoned test sets (for ASR evaluation).
                - Dataset: Clean test set (for CDA evaluation).
                - Dataset: Fine-tuning training set (used to train the student model).
                - Dataset: Fine-tuning validation set (used to evaluate during training).
        """

        # import dataset class
        dataset_cls = getattr(
            self._import_package(f"{config["modules"]["dataset"]["root"]}.{config["modules"]["dataset"]["file"]}"),
            config["modules"]["dataset"]["class"],
        )

        # initialize global train set
        global_train_set = dataset_cls(
            root=config["dataset"]["root"],
            train=True,
            transform=config["dataset"]["base_transform"],
            target_transform=config["dataset"]["base_target_transform"],
            download=config["dataset"]["download"],
        )

        finetune_indices = np.loadtxt(
            f"{config["checkpoint"]["root"]}/{config["checkpoint"]["finetune_subset"]}",
            delimiter=",",
            dtype=np.int32,
        )
        finetune_shuffled_indices = self.rng.permutation(finetune_indices)[: config["kd"]["finetune_subset_size"]]

        finetune_set = Subset(global_train_set, indices=finetune_shuffled_indices)
        f_trainset, f_valset = random_split(
            dataset=finetune_set,
            lengths=config["train"]["train_val_ratio"],
        )

        # initialize global base/clean test set (to measure cda metric)
        test_set_cda = dataset_cls(
            root=config["dataset"]["root"],
            train=False,
            transform=config["dataset"]["base_transform"],
            target_transform=config["dataset"]["base_target_transform"],
            download=config["dataset"]["download"],
        )

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

        # wrap test_set_cda with each trigger to create test_set_asr
        test_sets_asr = []
        for i, trigger in enumerate(triggers):

            clean_transform = (
                config["dataset"]["clean_transform"]
                if config["dataset"]["clean_transform"]
                else v2.Compose([v2.Identity()])
            )
            poison_transform = v2.Compose([trigger])

            # poison the test set (for measuring asr metric)
            poisoned_testset = PoisonedDatasetWrapper(
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

            # check if normalize is True then apply based on extracted mean and std
            if config["dataset"]["normalize"]:
                train_set_mean, train_set_std = ...
                clean_transform.transforms.append(v2.Normalize(train_set_mean, train_set_std))
                poison_transform.transforms.append(v2.Normalize(train_set_mean, train_set_std))

            test_sets_asr.append(poisoned_testset)

            # save trigger face as an image [debug]
            logger.save_trigger_pattern(
                path=config["logger"]["trigger"]["path"].format(i + 1),
                filename=config["logger"]["trigger"]["filename"],
                trigger_policy=trigger,
                bg_size=config["dataset"]["image_shape"],
                bg_color=config["logger"]["trigger"]["bg_color"],
                dataset=test_set_cda,
                n_samples=config["logger"]["trigger"]["n_samples"],
                clamp=config["logger"]["trigger"]["clamp"],
                show=config["logger"]["trigger"]["show"],
            )

        return test_sets_asr, test_set_cda, f_trainset, f_valset

    def _prepare_models(self) -> tuple[list[nn.Module], nn.Module, nn.Module | None]:
        """
        Prepare and initialize models for knowledge distillation.

        This includes:
        - Loading pretrained teacher models (one per subset).
        - Creating the teaching assistant (TA) model.
        - Optionally creating the student model (if enabled in config).

        Returns:
            tuple: A tuple containing:
                - list[nn.Module]: List of teacher models (frozen and in eval mode).
                - nn.Module: TA model (train mode).
                - nn.Module | None: Student model (train mode), or None if `only_ta` is enabled.
        """

        teacher_models = []

        # load sp models
        for i in range(config["dataset"]["num_subsets"]):

            if config["logger"]["weights"]["only_state_dict"]:
                with open(f"{config["checkpoint"]["root"]}/{config["checkpoint"]["model_dict"].format(i+1)}") as f:
                    teacher_model_dict = json.load(f)
                teacher_model_dict["weights"] = (
                    f"{config["checkpoint"]["root"]}/{config["checkpoint"]["model_weight"].format(i + 1)}"
                )

                teacher_model_cls = getattr(
                    self._import_package(f"{config["modules"]["model"]["root"]}.{teacher_model_dict["file"]}"),
                    teacher_model_dict["class"],
                )

                # initialize the model
                teacher_model = teacher_model_cls(
                    arch=teacher_model_dict["arch"],
                    in_channels=config["dataset"]["image_shape"][0],
                    num_classes=config["dataset"]["num_classes"],
                    weights=teacher_model_dict["weights"],
                    device=config["misc"]["device"],
                    verbose=config["misc"]["verbose"],
                )
            else:
                teacher_model = torch.load(
                    f"{config["checkpoint"]["root"]}/{config["checkpoint"]["model_weight"].format(i + 1)}",
                    map_location="cpu",
                )

            for param in teacher_model.parameters():
                param.requires_grad = False

            teacher_model.eval()
            teacher_models.append(teacher_model)

        # create ta model
        ta_model_cls = getattr(
            self._import_package(f"{config["modules"]["model"]["root"]}.{config["kd"]["ta"]["model"]["file"]}"),
            config["kd"]["ta"]["model"]["class"],
        )

        ta_model = ta_model_cls(
            arch=config["kd"]["ta"]["model"]["arch"],
            in_channels=config["dataset"]["image_shape"][0],
            num_classes=config["dataset"]["num_classes"],
            weights=config["kd"]["ta"]["model"]["weights"],
            device=config["misc"]["device"],
            verbose=config["misc"]["verbose"],
        )

        ta_model.train()

        # create student model
        if not config["kd"]["only_ta"]:
            student_model_cls = getattr(
                self._import_package(
                    f"{config["modules"]["model"]["root"]}.{config["kd"]["student"]["model"]["file"]}"
                ),
                config["kd"]["student"]["model"]["class"],
            )

            student_model = student_model_cls(
                arch=config["kd"]["student"]["model"]["arch"],
                in_channels=config["dataset"]["image_shape"][0],
                num_classes=config["dataset"]["num_classes"],
                weights=config["kd"]["student"]["model"]["weights"],
                device=config["misc"]["device"],
                verbose=config["misc"]["verbose"],
            )

            student_model.train()
        else:
            student_model = None

        return teacher_models, ta_model, student_model

    def _train_and_validate(
        self,
        stage_idx: int,
        train_set: Dataset,
        val_set: Dataset,
        teacher_models: list[nn.Module],
        student_model: nn.Module,
    ) -> None:
        """
        Train a student model via knowledge distillation and validate it over epochs.

        This method performs supervised training using multiple teacher models by
        averaging their outputs. The student model is trained to match the teacher
        outputs (soft labels) and ground-truth labels via a distillation loss. It
        logs metrics, saves model weights per epoch, and plots results.

        Args:
            stage_idx (int): Index of the current distillation stage (1 for TA, 2 for Student).
            train_set (Dataset): Training dataset used for distillation.
            val_set (Dataset): Validation dataset used to evaluate generalization.
            teacher_models (list[nn.Module]): List of pretrained, frozen teacher models.
            student_model (nn.Module): The model being trained through distillation.
        """
        # save stats per epoch
        train_loss_per_epoch, train_acc_per_epoch = [], []
        val_loss_per_epoch, val_acc_per_epoch = [], []

        # initialize data loaders
        train_loader = DataLoader(train_set, batch_size=config["train"]["train_batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config["train"]["val_batch_size"], shuffle=False)

        # initialize criterion, optimizer and lr_scheduler
        criterion = DistillationLoss(temperature=config["kd"]["ta"]["temprature"], alpha=config["kd"]["ta"]["alpha"])
        optimizer: optim.Optimizer = config["train"]["optimizer"](
            student_model.parameters(),
            **config["train"]["optimizer_params"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config["train"]["scheduler_params"],
        )

        # initialize train and validation accuracy metric
        train_acc_metric = MulticlassAccuracy(config["dataset"]["num_classes"]).to(config["misc"]["device"])
        val_acc_metric = MulticlassAccuracy(config["dataset"]["num_classes"]).to(config["misc"]["device"])

        epochs = config["kd"]["ta"]["epochs"]

        # store hyperparameters as a json file
        logger.save_hyperparameters(
            path=Path(config["logger"]["hyperparameters"]["path"].format(stage_idx)),
            filename=config["logger"]["hyperparameters"]["filename"],
            epochs=epochs,
            mean_per_channel="NotImplementedYet",
            std_per_channel="NotImplementedYet",
            criterion=criterion.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )

        for epoch in range(1, epochs + 1):
            # train phase
            student_model.train()
            train_loss = 0

            for x, y_true in train_loader:
                # move data to <device>
                x, y_true = x.to(config["misc"]["device"]), y_true.to(config["misc"]["device"])

                # forward and backward pass
                with torch.no_grad():
                    teacher_logits = sum(t(x) for t in teacher_models) / len(teacher_models)

                ta_logits = student_model(x)

                loss = criterion(ta_logits, teacher_logits, y_true)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # calculate and store metrics per batch
                train_loss += loss.item() * len(x)
                train_acc_metric.update(ta_logits, y_true)

            # calculate metrics per epoch
            train_loss /= len(train_loader.dataset)
            train_acc = train_acc_metric.compute().item()

            # store metrics per epoch
            train_acc_per_epoch.append(train_acc)
            train_loss_per_epoch.append(train_loss)

            # reset metrics for the next epoch
            train_acc_metric.reset()

            # validation phase
            student_model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y_true in val_loader:
                    # move data to <device>
                    x, y_true = x.to(config["misc"]["device"]), y_true.to(config["misc"]["device"])

                    # forward pass
                    teacher_logits = sum(t(x) for t in teacher_models) / len(teacher_models)
                    ta_logits = student_model(x)
                    loss = criterion(ta_logits, teacher_logits, y_true)

                    # calculate and store metrics per batch
                    val_loss += loss.item() * len(x)
                    val_acc_metric.update(ta_logits, y_true)

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
                path=Path(config["logger"]["metrics"]["train_path"].format(stage_idx)),
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
                path=Path(config["logger"]["weights"]["path"].format(stage_idx)),
                filename=config["logger"]["weights"]["filename"],
                model=student_model,
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
                path=Path(config["logger"]["plot_metrics"]["path"].format(stage_idx)),
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
                path=Path(config["logger"]["pred_demo"]["train_path"].format(stage_idx)),
                filename=data_role,
                model=student_model,
                dataset=data_value,
                nrows=config["logger"]["pred_demo"]["nrows"],
                ncols=config["logger"]["pred_demo"]["ncols"],
                save_grid=config["logger"]["pred_demo"]["save_grid"],
                show_grid=config["logger"]["pred_demo"]["show_grid"],
                clamp=config["logger"]["pred_demo"]["clamp"],
            )

    def _test(
        self,
        stage_idx: int,
        test_sets_asr: list[Dataset],
        test_set_cda: Dataset,
        student_model: nn.Module,
    ) -> None:
        """
        Evaluate a trained student model on poisoned and clean test sets.

        This function calculates:
        - ASR (Attack Success Rate) on multiple poisoned test datasets.
        - CDA (Clean Data Accuracy) on both poisoned and clean datasets.
        - Confusion matrices for detailed class-wise analysis.
        - Stores results in CSVs and optionally prints them.

        Args:
            stage_idx (int): Index of the distillation stage (used for logging paths).
            test_sets_asr (list[Dataset]): List of poisoned test datasets to evaluate ASR.
            test_set_cda (Dataset): Clean test dataset to evaluate CDA.
            student_model (nn.Module): The trained student model to be evaluated.
        """

        # initialize test asr (attack success rate) metric and move to <device>
        test_asr_metric = AttackSuccessRate(config["dataset"]["target_index"]).to(config["misc"]["device"])
        test_cda_metric = CleanDataAccuracy().to(config["misc"]["device"])

        # iterate over posioned test sets
        asr_per_dataset, cda_per_dataset = [], []
        for sp_idx, test_set_asr in enumerate(test_sets_asr):
            true_labels_asr, predictions_asr = [], []
            test_loader_asr = DataLoader(test_set_asr, batch_size=config["test"]["test_batch_size"], shuffle=False)

            with torch.no_grad():
                for x_asr, y_asr_true, poisoned_mask, y_raw in test_loader_asr:

                    # move data to <device>
                    x_asr, y_asr_true = x_asr.to(config["misc"]["device"]), y_asr_true.to(config["misc"]["device"])
                    y_raw = y_raw.to(config["misc"]["device"])

                    # forward
                    ta_logits = student_model(x_asr)

                    # calculate and store metrics per batch
                    test_asr_metric.update(ta_logits, poisoned_mask)
                    test_cda_metric.update(ta_logits, y_raw, poisoned_mask)

                    # extend true/pred labels
                    true_labels_asr.extend(y_raw.cpu())  # y_raw instead of y_asr_true for better demonstration
                    predictions_asr.extend(ta_logits.argmax(dim=1).cpu())

                # calculate and store metrics
                test_asr = test_asr_metric.compute().item()
                test_cda = test_cda_metric.compute().item()
                asr_per_dataset.append(test_asr)
                cda_per_dataset.append(test_cda)

                # reset metrics
                test_asr_metric.reset()
                test_cda_metric.reset()

            # print results in the standard output
            if config["misc"]["verbose"]:
                print(f"[Test] dataset_poison_sp{sp_idx+1} [asr: {test_asr*100:5.2f}% , cda: {test_cda*100:5.2f}%]")

            # store confusion matrix as a csv file
            true_labels_asr, predictions_asr = map(torch.tensor, [true_labels_asr, predictions_asr])
            confmat = MulticlassConfusionMatrix(config["dataset"]["num_classes"])
            cm = confmat(predictions_asr, true_labels_asr)
            logger.save_labeled_matrix(
                path=Path(f"{config['logger']['confusion_matrix']['path']}".format(stage_idx)),
                filename=f"poisones_dataset_{sp_idx+1}_{config['logger']['confusion_matrix']['filename']}",
                matrix=cm,
                row0_col0_title="True/Pred",  # Title for the top-left header cell
                row_labels=list(range(config["dataset"]["num_classes"])),  # Class labels
            )

        # clean test set
        # initialize test cda (clean data accuracy) metric and move to <device>
        test_cda_metric = CleanDataAccuracy().to(config["misc"]["device"])

        true_labels_cda, predictions_cda = [], []
        test_loader_cda = DataLoader(test_set_cda, batch_size=config["test"]["test_batch_size"], shuffle=False)

        with torch.no_grad():
            for x_cda, y_cda_true in test_loader_cda:

                # move data to <device>
                x_cda, y_cda_true = x_cda.to(config["misc"]["device"]), y_cda_true.to(config["misc"]["device"])

                # forward
                ta_logits = student_model(x_cda)

                # calculate and store metrics per batch
                test_cda_metric.update(ta_logits, y_cda_true, torch.ones(size=(len(x_cda),), dtype=torch.bool))

                # extend true/pred labels
                true_labels_cda.extend(y_cda_true.cpu())
                predictions_cda.extend(ta_logits.argmax(dim=1).cpu())

            # calculate and store metrics
            test_cda = test_cda_metric.compute().item()

            # reset metrics
            test_cda_metric.reset()

        # print results in the standard output
        if config["misc"]["verbose"]:
            print(f"[Test] dataset_clean [cda: {test_cda*100:5.2f}%]")

        # store confusion matrix as a csv file
        true_labels_cda, predictions_cda = map(torch.tensor, [true_labels_cda, predictions_cda])
        confmat = MulticlassConfusionMatrix(config["dataset"]["num_classes"])
        cm = confmat(predictions_cda, true_labels_cda)
        logger.save_labeled_matrix(
            path=Path(f"{config['logger']['confusion_matrix']['path']}".format(stage_idx)),
            filename=f"clean_dataset_{config['logger']['confusion_matrix']['filename']}",
            matrix=cm,
            row0_col0_title="True/Pred",  # Title for the top-left header cell
            row_labels=list(range(config["dataset"]["num_classes"])),  # Class labels
        )

        # store metrics as a csv file
        logger.save_metrics(
            path=Path(f"{config["logger"]["metrics"]["test_path"]}".format(stage_idx)),
            filename=config["logger"]["metrics"]["filename"],
            c_cda=test_cda,
            **{f"p_asr_{i+1}": asr_per_dataset[i] for i in range(len(test_sets_asr))},
            **{f"p_cda_{i+1}": cda_per_dataset[i] for i in range(len(test_sets_asr))},
        )

    def _analyze_feature_maps(
        self,
        stage_idx: int,
        test_sets_asr: list[Dataset],
        test_set_cda: Dataset,
        student_model: nn.Module,
    ) -> None:
        """
        Analyze and save feature maps from the student model for both poisoned and clean test samples.

        This function:
        - Extracts and stores feature maps for the first batch of each poisoned dataset.
        - Extracts and stores feature maps for the first batch of the clean dataset.
        - Supports configurable normalization, aggregation, and overview plotting for each layer.

        Args:
            stage_idx (int): Index of the current stage (used in output paths).
            test_sets_asr (list[Dataset]): List of poisoned test datasets.
            test_set_cda (Dataset): Clean test dataset.
            student_model (nn.Module): The trained student model to inspect.
        """

        student_model.eval()
        model_inspector = FeatureExtractor(student_model)
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
                path=f"{config["logger"]["feature_maps"]["path"].format(stage_idx)}/poisoned_dataset_{j+1}",
                feature_dict=feature_maps,
                normalize=config["logger"]["feature_maps"]["normalize"],
                aggregation=config["logger"]["feature_maps"]["aggregation"],
                overview=config["logger"]["feature_maps"]["overview"],
            )

        # [test set cda]
        # normalize (standardize) samples if needed
        # transform orders: [base_transforms - poison_transforms - v2.Normalize]
        # if config["dataset"]["normalize"]:
        #     if isinstance(test_set_cda.transform.transforms[-1], v2.Normalize):
        #         del test_set_cda.transform.transforms[-1]
        #     test_set_cda.transform.transforms.append(v2.Normalize(mean=self.mean_per_sp[i], std=self.std_per_sp[i]))

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
            path=f"{config["logger"]["feature_maps"]["path"].format(stage_idx)}/clean_dataset",
            feature_dict=feature_maps,
            normalize=config["logger"]["feature_maps"]["normalize"],
            aggregation=config["logger"]["feature_maps"]["aggregation"],
            overview=config["logger"]["feature_maps"]["overview"],
        )

    def _analyze_grad_cam(
        self,
        stage_idx: int,
        test_sets_asr: list[Dataset],
        test_set_cda: Dataset,
        student_model: nn.Module,
    ) -> None:
        """
        Generate and store Grad-CAM heatmaps for both poisoned and clean datasets.

        This method:
        - Applies Grad-CAM on the first batch of each poisoned test set to identify activated regions.
        - Overlays and saves heatmaps to visualize model attention.
        - Repeats the same process for the clean test set.

        Args:
            stage_idx (int): Index of the current training stage (used in logging paths).
            test_sets_asr (list[Dataset]): List of poisoned test datasets.
            test_set_cda (Dataset): Clean test dataset.
            student_model (nn.Module): The student model under evaluation.
        """

        student_model.eval()
        grad_cam = GradCAM(model=student_model, target_layers=config["logger"]["grad_cam"]["layers"])
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
                path=f"{config["logger"]["grad_cam"]["path"].format(stage_idx)}/poisoned_dataset_{j+1}",
                overlays_dict=overlays,
                normalize=config["logger"]["grad_cam"]["normalize"],
                overview=config["logger"]["grad_cam"]["overview"],
            )

        # [test set cda]
        # normalize (standardize) samples if needed
        # transform orders: [base_transforms - poison_transforms - v2.Normalize]
        # if config["dataset"]["normalize"]:
        #     if isinstance(test_set_cda.transform.transforms[-1], v2.Normalize):
        #         del test_set_cda.transform.transforms[-1]
        #     test_set_cda.transform.transforms.append(v2.Normalize(mean=self.mean_per_sp[i], std=self.std_per_sp[i]))

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
            path=f"{config["logger"]["grad_cam"]["path"].format(stage_idx)}/clean_dataset",
            overlays_dict=overlays,
            normalize=config["logger"]["grad_cam"]["normalize"],
            overview=config["logger"]["grad_cam"]["overview"],
        )
