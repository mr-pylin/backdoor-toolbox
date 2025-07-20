import torch
from torch import optim
from torchvision.transforms import v2
from backdoor_toolbox.triggers.target_transform import TargetTriggerTypes
from backdoor_toolbox.triggers.transform import TriggerTypes

__config = {
    "image_shape": (1, 28, 28),  # consider `image_shape` after passing through `v2.ToImage()`
    "target_index": 0,
}

config = {
    "dataset": {
        "root": "./data",
        "train": True,
        "normalize": True,  # mean and std is computed dynamically based on chosen dataset
        "clean_transform": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "clean_target_transform": None,
        "poisoned_transform": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                TriggerTypes.SOLID.value(__config["image_shape"], (1.0,), (4, 4), (22, 22)),
            ]
        ),
        "poisoned_target_transform": v2.Compose([TargetTriggerTypes.FLIPLABEL.value(__config["target_index"])]),
        "target_index": __config["target_index"],
        "victim_indices": (1, 2, 3, 4, 5, 6, 7, 8, 9),
        "poison_ratio": 0.01,
        "download": False,
        "num_classes": 10,
        "image_shape": __config["image_shape"],
    },
    "modules": {
        "dataset": {
            # from "root"."file" import "class"
            "root": "datasets",
            "file": "mnist",
            "clean_class": "CleanMNIST",
            "poisoned_class": "PoisonedMNIST",
        },
        "model": {
            # from "root"."file" import "class"
            "root": "models.cnn",
            "file": "resnet_wrapper",
            "class": "CustomResNet",
            "params": {
                "weights": None,  # e.g. "ResNet18_Weights.IMAGENET1K_V1" or path to .pth file
                "kwargs": {"model_name": "resnet18"},
            },
        },
    },
    "loss": {  # loss parameters for step 2: backdoor removal
        "epsilon_1": 0.3,
        "epsilon_2": 0.5,
        "lambda": 1.0,
    },
    "train_val": {
        "train_val_ratio": (0.9, 0.1),  # 0.9 for train set and 0.1 for validation set
        "step_1_epochs": 1,
        "step_2_epochs": 1,
        "train_batch_size": 64,
        "val_batch_size": 128,
        "optimizer": optim.Adam,
        "optimizer_params": {"lr": 0.01},
        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 2, "threshold": 1e-3},
    },
    "test": {"test_batch_size": 128},
    "log": {
        "root": "./logs/attack/qa",
        "include_date": True,
        "config": {"path": "src/backdoor_toolbox/routines/attacks/quantization_attack", "filename": "config"},
        "hyperparameters": {"path": "train_val", "filename": "hyperparameters"},
        "metrics": {"train_path": "train_val", "test_path": "test", "filename": "report"},
        "weights": {"path": "train_val/weights"},
        "plot": {
            "path": "train_val/plots",
            "save_format": "svg",
            "metrics": [
                {"filename": "loss", "ylabel": "Loss", "title": "Loss over time", "show": False},
                {"filename": "asr", "ylabel": "ASR", "title": "ASR over time", "show": False},
                {"filename": "cda", "ylabel": "CDA", "title": "CDA over time", "show": False},
            ],
        },
        "confusion_matrix": {"path": "test", "filename": "confusion_matrix"},
        "demo": {"train_path": "train_val/demo", "test_path": "test/demo", "nrows": 7, "ncols": 20, "show": False},
    },
    "misc": {
        "seed": 42,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "verbose": True,
    },
}
