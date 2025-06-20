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
        "model": {
            "root": "models.cnn",
        },
    },
    "checkpoint": {
        "root": "./logs/attack/multi_attack/2025-06-15-06-02-40",  # also update in config["logger"]["sub_root"]
        "model_dict": "sp{0}/model/config.json",
        "model_weight": "sp{0}/model/weights/model_epoch_15.pth",
        "hyperparameters": "sp{0}/train/hyperparameters.json",
        "finetune_subset": "finetune_subset_indices.csv",
    },
    "kd": {
        "finetune_subset_size": 6000,
        "only_ta": False,
        "ta": {
            "epochs": 15,
            "model": {
                "file": "resnet_wrapper",
                "class": "CustomResNet",
                "arch": "resnet34",
                "weights": None,
            },
            "temprature": 4.0,
            "alpha": 0.5,
        },
        "student": {
            "epochs": 15,
            "model": {
                "file": "resnet_wrapper",
                "class": "CustomResNet",
                "arch": "resnet18",
                "weights": None,
            },
            "temprature": 4.0,
            "alpha": 0.5,
        },
    },
    "dataset": {
        "num_subsets": 7,
        "root": "./data",
        "base_transform": v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        "base_target_transform": None,
        "download": False,
        "image_shape": (1, 28, 28),  # consider `image_shape` after passing through `v2.ToImage()`
        "clean_transform": None,
        "clean_target_transform": None,
        "poison_target_transform": v2.Compose([TargetTriggerTypes.FLIPLABEL.value(__config["target_index"])]),
        "target_index": __config["target_index"],
        "victim_indices": (1, 2, 3, 4, 5, 6, 7, 8, 9),
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
                r"./assets/blend_trigger/noise.jpg",
                r"./assets/blend_trigger/kitty.jpg",
                r"./assets/blend_trigger/pattern.jpg",
                r"./assets/blend_trigger/creeper.jpg",
                r"./assets/blend_trigger/chess.jpg",
            ],
        },
    },
    "train": {
        "train_val_ratio": (0.8, 0.2),  # (train, validation) split
        "train_batch_size": 64,
        "val_batch_size": 128,
        "optimizer": optim.Adam,
        "optimizer_params": {"lr": 0.01},
        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 2, "threshold": 1e-3},
    },
    "test": {
        "test_batch_size": 128,
    },
    "logger": {
        "root": "./logs/defense/knowledge_distillation",
        "sub_root": "2025-06-15-06-02-40",
        "config": {
            "src_path": "src/backdoor_toolbox/routines/defenses/ensemble_learning",
            "dst_path": "",
            "filename": "config",
        },
        "hyperparameters": {
            "path": "stage{0}/train",
            "filename": "hyperparameters",
        },
        "metrics": {
            "train_path": "stage{0}/train/metrics",
            "test_path": "stage{0}/test/metrics",
            "filename": "report",
        },
        "weights": {
            "path": "stage{0}/model/weights",
            "filename": "model",
            "only_state_dict": True,
        },
        "plot_metrics": {
            "path": "stage{0}/train/metrics",
            "save_format": "svg",
            "show": False,
            "markers": True,
            "metrics": [
                {
                    "filename": "loss",
                    "ylabel": "Loss",
                    "title": "Loss over time",
                },
                {
                    "filename": "accuracy",
                    "ylabel": "Accurace",
                    "title": "Accuracy over time",
                },
            ],
        },
        "confusion_matrix": {
            "path": "stage{0}/test/metrics",
            "filename": "confusion_matrix",
        },
        "pred_demo": {
            "train_path": "stage{0}/train/demo",
            "test_path": "stage{0}/test/demo",
            "nrows": 8,
            "ncols": 24,
            "clamp": True,
            "save_grid": True,
            "show_grid": False,
        },
        "feature_maps": {
            "path": "stage{0}/analysis/feature_maps",
            "num_images": 4,
            "layers": ["layer1", "layer2", "layer3"],
            "normalize": True,
            "aggregation": True,
            "overview": True,
        },
        "grad_cam": {
            "path": "stage{0}/analysis/grad_cam",
            "num_images": 36,
            "layers": ["layer3"],
            "normalize": True,
            "overview": True,
        },
    },
    "misc": {
        "seed": 0,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "verbose": True,
    },
}
