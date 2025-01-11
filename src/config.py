import torch
from torch import optim
from torchvision.transforms import v2

# from triggers.target_transform import TargetTriggerTypes
# from triggers.transform import TriggerTypes


# specify the routine
routine_dict = {
    # e.g. from "root"."file" import "class"
    "root": "routines",
    "file": "clean",
    "class": "CleanRoutine",
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
        "root": "./logs/log",
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
        "confusion_matrix": {"path": "test", "filename": "confusion_matrix"},
    },
    "misc": {
        "seed": 42,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    },
}


# configurations for backdoor attack
attack_config = {
    "dataset": {
        "root": "./data",
        "train": True,
        "transform": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                # v2.Normalize()
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
        "root": "./logs/log",
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
        "confusion_matrix": {"path": "test", "filename": "confusion_matrix"},
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

# METHOD_MODULE_ROOT = "routines"
# METHOD_MODULE_NAME = "clean"
# CLASS_NAME = "CleanNet"
# TYPE_VERBOSE = True

# TRIGGER = TriggerTypes.SOLID.value
# TRIGGER_IMAGE_SHAPE = (1, 28, 28)
# TRIGGER_COLOR = (1.0,)
# TRIGGER_SIZE = (6, 6)
# TRIGGER_POSITION = (20, 20)
# TARGET_TRIGGER = TargetTriggerTypes.FLIPLABEL.value
# TARGET_INDEX = 0

# DATASET_ROOT = "./data"
# DATASET_MODULE_ROOT = "datasets"
# DATASET_MODULE_NAME = "mnist"
# CLEAN_CLASS = "CleanMNIST"
# POISONED_CLASS = "PoisonedMNIST"
# TRAIN_FLAG = True
# DOWNLOAD_FLAG = False
# NUM_CLASSES = 10
# CLEAN_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
# CLEAN_TARGET_TRANSFORM = None
# VICTIM_INDICES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# POISON_RATIO = 0.05
# POISONED_TRANSFORM = v2.Compose(
#     [
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True),
#         TRIGGER(TRIGGER_IMAGE_SHAPE, TRIGGER_COLOR, TRIGGER_SIZE, TRIGGER_POSITION),
#     ]
# )
# POISONED_TARGET_TRANSFORM = v2.Compose([TARGET_TRIGGER(TARGET_INDEX)])

# MODEL_MODULE_ROOT = "models.cnn"
# MODEL_MODULE_NAME = "resnet18"
# MODEL_CLASS = "ResNet18"
# MODEL_WEIGHTS = None  # "ResNet18_Weights.IMAGENET1K_V1"

# MISSING_VAL = True
# TRAIN_VAL_RATIO = (0.9, 0.1)
# EPOCHS = 10
# TRAIN_BATCH_SIZE = 64
# VAL_BATCH_SIZE = 64
# TEST_BATCH_SIZE = 64
# INITIAL_LR = 0.01
# OPTIMIZER = optim.Adam

# LOG_ROOT = "./logs/log"
# LOG_INCLUDE_DATE = True
# LOG_VERBOSE = True

# SEED = 42
# DEVICE = torch.device("cuda")

# configs = {
#     "type": {
#         "type": TYPE,
#         "module_root": METHOD_MODULE_ROOT,
#         "module_name": METHOD_MODULE_NAME,
#         "class_name": CLASS_NAME,
#         "verbose": TYPE_VERBOSE,
#     },
#     "trigger": {
#         "trigger": TRIGGER,
#         "trigger_params": {
#             "image_shape": TRIGGER_IMAGE_SHAPE,
#             "color": TRIGGER_COLOR,
#             "size": TRIGGER_SIZE,
#             "position": TRIGGER_POSITION,
#         },
#         "target_trigger": TARGET_TRIGGER,
#         "target_trigger_params": {"target_index": TARGET_INDEX},
#     },
#     "dataset": {
#         "dataset_root": DATASET_ROOT,
#         "module_root": DATASET_MODULE_ROOT,
#         "module_name": DATASET_MODULE_NAME,
#         "clean_class": CLEAN_CLASS,
#         "poisoned_class": POISONED_CLASS,
#         "train_flag": TRAIN_FLAG,
#         "download_flag": DOWNLOAD_FLAG,
#         "num_classes": NUM_CLASSES,
#         "clean_transform": CLEAN_TRANSFORM,
#         "clean_target_transform": CLEAN_TARGET_TRANSFORM,
#         "victim_indices": VICTIM_INDICES,
#         "poison_ratio": POISON_RATIO,
#         "poisoned_transform": POISONED_TRANSFORM,
#         "poisoned_target_transform": POISONED_TARGET_TRANSFORM,
#     },
#     "model": {
#         "module_root": MODEL_MODULE_ROOT,
#         "module_name": MODEL_MODULE_NAME,
#         "model_class": MODEL_CLASS,
#         "weights": MODEL_WEIGHTS,
#         "num_classes": NUM_CLASSES,
#     },
#     "train": {
#         "missing_val": MISSING_VAL,
#         "train_val_ratio": TRAIN_VAL_RATIO,
#         "epochs": EPOCHS,
#         "train_batch_size": TRAIN_BATCH_SIZE,
#         "val_batch_size": VAL_BATCH_SIZE,
#         "test_batch_size": TEST_BATCH_SIZE,
#         "initial_lr": INITIAL_LR,
#         "optimizer": OPTIMIZER,
#         "optimizer_params": {"lr": INITIAL_LR},
#     },
#     "log": {
#         "root": LOG_ROOT,
#         "include_date": LOG_INCLUDE_DATE,
#         "verbose": LOG_VERBOSE,
#     },
#     "misc": {
#         "seed": SEED,
#         "device": DEVICE,
#     },
# }
