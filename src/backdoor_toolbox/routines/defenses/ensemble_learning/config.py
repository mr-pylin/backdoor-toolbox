import torch
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
        "root": "./logs/attack/multi_attack/MNIST/2025-07-02-07-12-23/2025-07-14-23-19-53",  # also update in config["logger"]["sub_root"]
        "model_dict": "sp{0}/model/config.json",
        "model_weight": "sp{0}/model/weights/model_epoch_15.pth",
        "hyperparameters": "sp{0}/train/hyperparameters.json",
    },
    "ensemble": {
        "vote": "hard",
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
        "normalize": False,  # whether to use v2.Normalize() or not [current implementation is inconsistent with normalization!]
        "num_classes": 10,
    },
    "trigger": {
        "triggers_cls": (
            TriggerTypes.SOLID.value,
            TriggerTypes.PATTERN.value,
            TriggerTypes.BLEND.value,
            TriggerTypes.NOISE.value,
        ),
        "blend": {
            "bg_paths": [
                # r"./assets/blend_trigger/noise.jpg",
                r"./assets/blend_trigger/chess.jpg",
                # r"./assets/blend_trigger/pattern.jpg",
                r"./assets/blend_trigger/kitty.jpg",
                r"./assets/blend_trigger/creeper.jpg",
            ],
        },
        "num_similarity": 0,
        "similarity_ratio": 0,
    },
    # check ./models/ for available models
    "test": {
        "test_batch_size": 128,
    },
    "logger": {
        "root": "./logs/defense/ensemble_learning",
        "sub_root": "MNIST/2025-07-14-23-19-53",
        "config": {
            "src_path": "src/backdoor_toolbox/routines/defenses/ensemble_learning",
            "dst_path": "",
            "filename": "config",
        },
        "trigger": {
            "path": "trigger/sp{}",
            "filename": "pattern_demo",
            "bg_color": 0.0,
            "n_samples": 16,  # if using the dataset to generate samples
            "clamp": False,
            "show": False,
        },
        "metrics": {
            "test_path": "test/metrics",
            "filename": "report",
        },
        "weights": {
            "only_state_dict": True,
        },
        "confusion_matrix": {
            "path": "test/metrics",
            "filename": "confusion_matrix",
        },
    },
    "misc": {
        "seed": 0,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "verbose": True,
    },
}
