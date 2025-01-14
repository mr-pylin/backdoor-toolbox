import torch
from torch import optim
from torchvision.transforms import v2

from triggers.target_transform import TargetTriggerTypes
from triggers.transform import TriggerTypes


# routine for clean
# routine_dict = {
#     # e.g. from "root"."file" import "class"
#     "root": "routines",
#     "file": "clean",
#     "class": "CleanRoutine",
#     "verbose": True,
# }

# routine for attack
routine_dict = {
    # e.g. from "root"."file" import "class"
    "root": "routines.attacks",
    "file": "quantization_attack",
    "class": "QuantizationAttackRoutine",
    "verbose": True,
}

clean_config = {
    "dataset": {
        "root": "./data",
        "train": True,
        "transform": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.1352,), std=(0.3088,)),
            ]
        ),
        "target_transform": None,
        "download": False,
        "num_classes": 10,
        "image_shape": (1, 28, 28),  # consider `image_shape` after passing through `v2.ToImage()`
    },
    "model": {
        "weights": None,  # e.g. "ResNet18_Weights.IMAGENET1K_V1"
    },
    "modules": {
        "dataset": {
            # e.g. from "root"."file" import "class"
            "root": "datasets",
            "file": "mnist",
            "class": "CleanMNIST",
        },
        "model": {
            # e.g. from "root"."file" import "class"
            "root": "models.cnn",
            "file": "resnet18",
            "class": "ResNet18",
        },
    },
    "train": {
        "train_val_ratio": (0.9, 0.1),  # 0.9 for train set and 0.1 for validation set
        "epochs": 15,
        "train_batch_size": 64,
        "val_batch_size": 128,
        "optimizer": optim.Adam,
        "optimizer_params": {
            "lr": 0.01,
        },
        "scheduler_params": {
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "threshold": 1e-3,
        },
    },
    "test": {
        "test_batch_size": 128,
    },
    "log": {
        "root": "./logs/clean",
        "include_date": True,
        "verbose": True,
        "hyperparameters": {
            "path": "train-val",
            "filename": "hyperparameters",
        },
        "metrics": {
            "train_path": "train-val",
            "test_path": "test",
            "filename": "report",
        },
        "weights": {
            "path": "train-val/weights",
        },
        "plot": {
            "path": "train-val/plots",
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
            "train_path": "train-val/demo",
            "test_path": "test/demo",
            "nrows": 7,
            "ncols": 20,
            "show": False,
        },
    },
    "misc": {
        "seed": 42,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    },
}


__attack_config = {
    "image_shape": (1, 28, 28),  # consider `image_shape` after passing through `v2.ToImage()`
    "target_index": 0,
}

attack_config = {
    "dataset": {
        "root": "./data",
        "train": True,
        "clean_transform": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.1352,), std=(0.3088,)),
            ]
        ),
        "clean_target_transform": None,
        "poisoned_transform": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                TriggerTypes.SOLID.value(__attack_config["image_shape"], (1.0,), (4, 4), (22, 22)),
                v2.Normalize(mean=(0.1352,), std=(0.3088,)),
            ]
        ),
        "poisoned_target_transform": v2.Compose([TargetTriggerTypes.FLIPLABEL.value(__attack_config["target_index"])]),
        "target_index": __attack_config["target_index"],
        "victim_indices": (1, 2, 3, 4, 5, 6, 7, 8, 9),
        "poison_ratio": 0.01,
        "download": False,
        "num_classes": 10,
        "image_shape": __attack_config["image_shape"],
    },
    "model": {
        "weights": None,  # e.g. "ResNet18_Weights.IMAGENET1K_V1"
    },
    "modules": {
        "dataset": {
            # e.g. from "root"."file" import "class"
            "root": "datasets",
            "file": "mnist",
            "clean_class": "CleanMNIST",
            "poisoned_class": "PoisonedMNIST",
        },
        "model": {
            # e.g. from "root"."file" import "class"
            "root": "models.cnn",
            "file": "resnet18",
            "class": "ResNet18",
        },
    },
    "train": {
        "train_val_ratio": (0.9, 0.1),  # 0.9 for train set and 0.1 for validation set
        "epochs": 15,
        "train_batch_size": 64,
        "val_batch_size": 128,
        "optimizer": optim.Adam,
        "optimizer_params": {
            "lr": 0.01,
        },
        "scheduler_params": {
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "threshold": 1e-3,
        },
    },
    "test": {
        "test_batch_size": 128,
    },
    "log": {
        "root": "./logs/qa",
        "include_date": True,
        "verbose": True,
        "hyperparameters": {
            "path": "train-val",
            "filename": "hyperparameters",
        },
        "metrics": {
            "train_path": "train-val",
            "test_path": "test",
            "filename": "report",
        },
        "weights": {
            "path": "train-val/weights",
        },
        "plot": {
            "path": "train-val/plots",
            "save_format": "svg",
            "metrics": [
                {
                    "filename": "loss",
                    "ylabel": "Loss",
                    "title": "Loss over time",
                    "show": False,
                },
                {
                    "filename": "asr",
                    "ylabel": "ASR",
                    "title": "ASR over time",
                    "show": False,
                },
                {
                    "filename": "cda",
                    "ylabel": "CDA",
                    "title": "CDA over time",
                    "show": False,
                },
            ],
        },
        "confusion_matrix": {
            "path": "test",
            "filename": "confusion_matrix",
        },
        "demo": {
            "train_path": "train-val/demo",
            "test_path": "test/demo",
            "nrows": 7,
            "ncols": 20,
            "show": False,
        },
    },
    "misc": {
        "seed": 42,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    },
}

# configurations for backdoor defense
defense_config = {}

# all configurations together
configs = {
    "clean": clean_config,
    "attack": attack_config,
    "defense": defense_config,
}
