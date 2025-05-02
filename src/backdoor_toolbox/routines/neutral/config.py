import torch
from torch import optim
from torchvision.transforms import v2


config = {
    # `from root.file import class`
    "modules": {
        # check ./datasets/ for available datasets
        "dataset": {
            "root": "datasets",
            "file": "mnist",
            "class": "MNIST",
        },
        # check ./models/ for available models
        "model": {
            "root": "models.cnn",
            "file": "resnet_wrapper",
            "class": "CustomResNet",
            "params": {
                "arch": "resnet18",
                "weights": None,  # options: {None, True, ".pth file"}
            },
        },
    },
    "dataset": {
        "root": "./data",
        "transform": v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        "target_transform": None,
        "download": False,
        "image_shape": (1, 28, 28),  # consider `image_shape` after passing through `v2.ToImage()`
        "normalize": True,  # whether to use v2.Normalize() or not
        "num_classes": 10,
    },
    "train_val": {
        "train_test_ratio": (0.8, 0.2),  # (train, test) split
        "train_val_ratio": (0.8, 0.2),  # (train, validation) split
        "epochs": 15,
        "train_batch_size": 64,
        "val_batch_size": 128,
        "optimizer": optim.Adam,
        "optimizer_params": {"lr": 0.01},
        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 2, "threshold": 1e-3},
    },
    "test": {
        "test_batch_size": 128,
    },
    "log": {
        "root": "./logs/neutral/neutral",
        "include_date": True,
        "config": {
            "path": "src/backdoor_toolbox/routines/neutral",
            "filename": "config",
        },
        "hyperparameters": {
            "path": "train_val",
            "filename": "hyperparameters",
        },
        "metrics": {
            "train_path": "train_val",
            "test_path": "test",
            "filename": "report",
        },
        "weights": {
            "path": "train_val/weights",
            "only_state_dict": True,
        },
        "plot": {
            "path": "train_val/plots",
            "save_format": "svg",
            "metrics": [
                {
                    "filename": "loss",
                    "ylabel": "Loss",
                    "title": "Loss over time",
                    "show": False,
                },
                {
                    "filename": "accuracy",
                    "ylabel": "Accurace",
                    "title": "Accuracy over time",
                    "show": False,
                },
            ],
        },
        "confusion_matrix": {
            "path": "test",
            "filename": "confusion_matrix",
        },
        "demo": {
            "train_path": "train_val/demo",
            "test_path": "test/demo",
            "nrows": 8,
            "ncols": 24,
            "show": False,
        },
    },
    "misc": {
        "seed": 42,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "verbose": True,
    },
}
