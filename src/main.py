import torch
import importlib

from configs import configs
from torch.utils.data import random_split, DataLoader

# from experiments.clean import CleanNet
# from models.cnn import resnet18
from utils.logger import Logger
from torchvision import models

# from utils.metrics import calculate_metrics

# import proper datasets
dataset_module = importlib.import_module(f"{configs["dataset"]["module_root"]}.{configs["dataset"]["module_name"]}")
model_module = importlib.import_module(f"{configs["model"]["module_root"]}.{configs["model"]["module_name"]}")
routine_module = importlib.import_module(f"{configs["type"]["module_root"]}.{configs["type"]["module_name"]}")


def main():
    # initialize logger
    logger = Logger(
        root=configs["log"]["root"],
        include_date=configs["log"]["include_date"],
        verbose=configs["log"]["verbose"],
    )

    # train a clean model
    if configs["type"]["type"] == "clean":

        # initialize datasets
        dataset_clean = getattr(dataset_module, configs["dataset"]["clean_class"])
        trainset = dataset_clean(
            root=configs["dataset"]["dataset_root"],
            train=configs["dataset"]["train_flag"],
            image_transform=configs["dataset"]["clean_transform"],
            image_target_transform=configs["dataset"]["clean_target_transform"],
            download=configs["dataset"]["download_flag"],
            seed=configs["misc"]["seed"],
        )

        if configs["train"]["missing_val"]:
            trainset, valset = random_split(trainset, configs["train"]["train_val_ratio"])
        else:
            pass

        testset = dataset_clean(
            root=configs["dataset"]["dataset_root"],
            train=not configs["dataset"]["train_flag"],
            image_transform=configs["dataset"]["clean_transform"],
            image_target_transform=configs["dataset"]["clean_target_transform"],
            download=configs["dataset"]["download_flag"],
            seed=configs["misc"]["seed"],
        )

        # initialize dataloaders
        # train_loader = DataLoader(trainset, batch_size=configs["train"]["train_batch_size"], shuffle=True)
        # val_loader = DataLoader(valset, batch_size=configs["train"]["val_batch_size"], shuffle=False)
        # test_loader = DataLoader(testset, batch_size=configs["train"]["test_batch_size"], shuffle=False)

        # initialize model
        model_class = getattr(model_module, configs["model"]["model_class"])
        model = model_class(
            weights=eval(f"models.{configs["model"]["weights"]}") if configs["model"]["weights"] else None,
            in_features=configs["trigger"]["trigger_params"]["image_shape"][0],
            num_classes=configs["model"]["num_classes"],
        ).to(configs["misc"]["device"])

        # initialize train, val, test routines
        routine_class = getattr(routine_module, configs["type"]["class_name"])
        routine = routine_class(
            model=model,
            config=configs,
            logger=logger,
            verbose=configs["type"]["verbose"],
        )

        # train and validation
        routine.train_and_validate(train_dataset=trainset, val_dataset=valset)

        # test
        routine.test(test_dataset=testset)


if __name__ == "__main__":
    main()
