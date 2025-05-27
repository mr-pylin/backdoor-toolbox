import copy
import json
import random
from pathlib import Path
from typing import Callable, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision.io import read_image
from torchvision.transforms import functional as tf
from torchvision.transforms import v2

from backdoor_toolbox.routines.defenses.ensemble_learning.config import config
from backdoor_toolbox.routines.base import BaseRoutine
from backdoor_toolbox.triggers.transform.transform import TriggerSelector
from backdoor_toolbox.utils.dataset import PoisonedDatasetWrapper
from backdoor_toolbox.utils.logger import Logger
from backdoor_toolbox.utils.metrics import AttackSuccessRate, CleanDataAccuracy

VotingType = Literal["hard", "soft"]

# instantiate a logger to save parameters, plots, weights, ...
logger = Logger(root=config["logger"]["root"], verbose=config["misc"]["verbose"])


class EnsembleLearningRoutine(BaseRoutine):
    def __init__(self):

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
        # prepare test sets for each service provider including both cda and asr test sets
        test_sets_asr, test_set_cda = self._prepare_data()

        # prepare model and initialize weights
        sp_models = self._prepare_model()

        self._test(test_sets_asr, test_set_cda, sp_models)

    def _prepare_data(self) -> tuple[Dataset, list[Dataset]]:

        # import dataset class
        dataset_cls = getattr(
            self._import_package(f"{config["modules"]["dataset"]["root"]}.{config["modules"]["dataset"]["file"]}"),
            config["modules"]["dataset"]["class"],
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
        blend_images = []
        for blend_image_path in config["trigger"]["blend"]["bg_paths"]:
            blend_img = read_image(blend_image_path)

            # convert image to grayscale if the dataset is in grayscale
            if config["dataset"]["image_shape"][0] == 1:
                blend_img = tf.rgb_to_grayscale(blend_img)

            # apply base transforms to the images
            blend_img = config["dataset"]["base_transform"](blend_img)
            blend_images.append(blend_img)

        # construct N random trigger transform policies
        trigger_selector = TriggerSelector(
            image_shape=config["dataset"]["image_shape"],
            trigger_types=config["trigger"]["triggers_cls"],
            num_triggers=config["dataset"]["num_subsets"],
            blend_images=blend_images,
            seed=config["misc"]["seed"],
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

        return test_sets_asr, test_set_cda

    def _prepare_model(self) -> list[nn.Module]:
        models = []

        # load sp models
        for i in range(config["dataset"]["num_subsets"]):

            if config["logger"]["weights"]["only_state_dict"]:
                with open(f"{config["checkpoint"]["root"]}/{config["checkpoint"]["model_dict"].format(i+1)}") as f:
                    model_dict = json.load(f)
                model_dict["weights"] = (
                    f"{config["checkpoint"]["root"]}/{config["checkpoint"]["model_weight"].format(i + 1)}"
                )

                model_cls = getattr(
                    self._import_package(f"{config["modules"]["model"]["root"]}.{model_dict["file"]}"),
                    model_dict["class"],
                )

                # initialize the model
                model = model_cls(
                    arch=model_dict["arch"],
                    in_channels=config["dataset"]["image_shape"][0],
                    num_classes=config["dataset"]["num_classes"],
                    weights=model_dict["weights"],
                    device=config["misc"]["device"],
                    verbose=config["misc"]["verbose"],
                )
            else:
                model = torch.load(
                    f"{config["checkpoint"]["root"]}/{config["checkpoint"]["model_weight"].format(i + 1)}",
                    map_location="cpu",
                )

            model.eval()
            models.append(model)

        return models

    def _test(self, test_sets_asr, test_set_cda, models) -> None:

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
                    all_outputs = []
                    for model in models:
                        all_outputs.append(model(x_asr))

                    y_asr_pred = torch.stack(all_outputs)  # shape: [num_models, batch_size, num_classes]

                    if config["ensemble"]["vote"] == "soft":
                        y_asr_pred = y_asr_pred.mean(dim=0)

                    elif config["ensemble"]["vote"] == "hard":
                        class_pred = torch.argmax(y_asr_pred, dim=-1)
                        one_hot_pred = torch.nn.functional.one_hot(class_pred, num_classes=y_asr_pred.shape[-1]).float()
                        y_asr_pred = one_hot_pred.mean(dim=0)
                        # y_asr_pred = torch.log(one_hot_pred.mean(dim=0) + 1e-10)  # Convert probs to logits

                    # calculate and store metrics per batch
                    test_asr_metric.update(y_asr_pred, poisoned_mask)
                    test_cda_metric.update(y_asr_pred, y_raw, poisoned_mask)

                    # extend true/pred labels
                    true_labels_asr.extend(y_raw.cpu())  # y_raw instead of y_asr_true for better demonstration
                    predictions_asr.extend(y_asr_pred.argmax(dim=1).cpu())

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
                path=Path(f"{config['logger']['confusion_matrix']['path']}".format(sp_idx)),
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
                all_outputs = []
                for model in models:
                    all_outputs.append(model(x_cda))

                y_cda_pred = torch.stack(all_outputs)  # shape: [num_models, batch_size, num_classes]

                if config["ensemble"]["vote"] == "soft":
                    y_cda_pred = y_cda_pred.mean(dim=0)

                elif config["ensemble"]["vote"] == "hard":
                    class_pred = torch.argmax(y_cda_pred, dim=-1)
                    one_hot_pred = torch.nn.functional.one_hot(class_pred, num_classes=y_cda_pred.shape[-1]).float()
                    y_cda_pred = one_hot_pred.mean(dim=0)
                    # y_cda_pred = torch.log(one_hot_pred.mean(dim=0) + 1e-10)  # Convert probs to logits

                # calculate and store metrics per batch
                test_cda_metric.update(y_cda_pred, y_cda_true, torch.ones(size=(len(x_cda),), dtype=torch.bool))

                # extend true/pred labels
                true_labels_cda.extend(y_cda_true.cpu())
                predictions_cda.extend(y_cda_pred.argmax(dim=1).cpu())

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
            path=Path(f"{config['logger']['confusion_matrix']['path']}".format(sp_idx)),
            filename=f"clean_dataset_{config['logger']['confusion_matrix']['filename']}",
            matrix=cm,
            row0_col0_title="True/Pred",  # Title for the top-left header cell
            row_labels=list(range(config["dataset"]["num_classes"])),  # Class labels
        )

        # store metrics as a csv file
        logger.save_metrics(
            path=Path(f"{config["logger"]["metrics"]["test_path"]}"),
            filename=config["logger"]["metrics"]["filename"],
            c_cda=test_cda,
            **{f"p_asr_{i+1}": asr_per_dataset[i] for i in range(len(test_sets_asr))},
            **{f"p_cda_{i+1}": cda_per_dataset[i] for i in range(len(test_sets_asr))},
        )

        # store and/or plot demo images with true and predicted labels
        # logger.save_image_predictions(
        #     path=Path(f"{config["logger"]["pred_demo"]["test_path"]}".format(sp_idx)),
        #     filename="test_cda_hard_vote",
        #     model=model,
        #     dataset=test_set_cda,
        #     nrows=config["logger"]["pred_demo"]["nrows"],
        #     ncols=config["logger"]["pred_demo"]["ncols"],
        #     save_grid=config["logger"]["pred_demo"]["save_grid"],
        #     show_grid=config["logger"]["pred_demo"]["show_grid"],
        #     clamp=config["logger"]["pred_demo"]["clamp"],
        # )
