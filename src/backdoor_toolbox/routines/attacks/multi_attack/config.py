import torch
from torch import optim
from torchvision.transforms import v2
from backdoor_toolbox.triggers.target_transform import TargetTriggerTypes
from backdoor_toolbox.triggers.transform.transform import TriggerTypes

__config = {
    "target_index": 0,
}

config = {
    # `from root.file import class`
    "modules": {
        # check ./datasets/ for available datasets
        "dataset": {
            "root": "datasets",
            "file": "mnist",
            "class": "MNIST",
        },
    },
    "dataset": {
        "num_subsets": 7,
        "extract_finetune_subset": True,
        "root": "./data",
        "base_transform": v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        "base_target_transform": None,
        "download": False,
        "subset_ratio": 0.4,
        "subset_overlap": True,
        "image_shape": (1, 28, 28),  # consider `image_shape` after passing through `v2.ToImage()`
        "clean_transform": None,
        "clean_target_transform": None,
        "poison_target_transform": v2.Compose([TargetTriggerTypes.FLIPLABEL.value(__config["target_index"])]),
        "target_index": __config["target_index"],
        "victim_indices": (1, 2, 3, 4, 5, 6, 7, 8, 9),
        "poison_ratio": 0.01,
        "normalize": False,  # whether to use v2.Normalize() or not [current implementation is inconsistentwith normalization!]
        "num_classes": 10,
    },
    "trigger": {
        "triggers_cls": (
            TriggerTypes.SOLID.value,
            TriggerTypes.PATTERN.value,
            TriggerTypes.NOISE.value,
            TriggerTypes.BLEND.value,
        ),
        "blend": {
            "bg_paths": [
                r"./data/blend_trigger/noise.jpg",
                r"./data/blend_trigger/kitty.jpg",
            ],
        },
    },
    # check ./models/ for available models
    "model": {
        "root": "models.cnn",
        "same": True,
        "random": False,
        "if_random": {
            "resnet": {
                "file": "resnet_wrapper",
                "class": "CustomResNet",
                "archs": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                "weights": None,
            },
            # "vggnet": {},
        },
        "else": {
            "file": "resnet_wrapper",
            "class": "CustomResNet",
            "arch": "resnet18",
            "weights": None,
        },
        "extract_configuration": True,
    },
    "train_val": {
        "train_test_ratio": (0.8, 0.2),  # (train, test) split
        "train_val_ratio": (0.8, 0.2),  # (train, validation) split
        "epochs": 7,
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
        "root": "./logs/attack/multi_attack",
        "include_date": True,
        "config": {
            "path": "src/backdoor_toolbox/routines/attacks/multi_attack",
            "filename": "config",
        },
        "hyperparameters": {
            "path": "sp{0}/train_val",
            "filename": "hyperparameters",
        },
        "metrics": {
            "train_path": "sp{0}/train_val",
            "test_path": "sp{0}/test",
            "filename": "report",
        },
        "weights": {
            "path": "sp{0}/train_val/weights",
            "only_state_dict": True,
        },
        "plot": {
            "path": "sp{0}/train_val/plots",
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
            "path": "sp{0}/test",
            "filename": "confusion_matrix",
        },
        "demo": {
            "train_path": "sp{0}/train_val/demo",
            "test_path": "sp{0}/test/demo",
            "nrows": 8,
            "ncols": 24,
            "show": False,
        },
    },
    "misc": {
        "seed": 0,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "verbose": True,
    },
}
