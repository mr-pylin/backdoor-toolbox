import torch
from torch import optim
from torchvision.transforms import v2


config = {
    "dataset": {
        "root": "./data",
        "train": True,
        "normalize": True,  # mean and std is computed dynamically based on chosen dataset
        "transform": v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        "target_transform": None,
        "download": False,
        "num_classes": 10,
        "image_shape": (1, 28, 28),  # consider `image_shape` after passing through `v2.ToImage()`
    },
    "modules": {
        "dataset": {
            # from "root"."file" import "class"
            "root": "datasets",
            "file": "mnist",
            "class": "CleanMNIST",
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
    "train_val": {
        "train_val_ratio": (0.9, 0.1),  # 0.9 for train set and 0.1 for validation set
        "epochs": 15,
        "train_batch_size": 64,
        "val_batch_size": 128,
        "optimizer": optim.Adam,
        "optimizer_params": {"lr": 0.01},
        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 2, "threshold": 1e-3},
    },
    "test": {"test_batch_size": 128},
    "log": {
        "root": "./logs/neutral",
        "include_date": True,
        "config": {"path": "src/backdoor_toolbox/routines/neutral", "filename": "config"},
        "hyperparameters": {"path": "train_val", "filename": "hyperparameters"},
        "metrics": {"train_path": "train_val", "test_path": "test", "filename": "report"},
        "weights": {"path": "train_val/weights"},
        "plot": {
            "path": "train_val/plots",
            "save_format": "svg",
            "metrics": [
                {"filename": "loss", "ylabel": "Loss", "title": "Loss over time", "show": False},
                {"filename": "accuracy", "ylabel": "Accurace", "title": "Accuracy over time", "show": False},
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
