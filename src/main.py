import importlib

from config import configs, routine_dict
from utils.logger import Logger


def main():

    # initialize the routine
    module_path = f"{routine_dict["root"]}.{routine_dict["file"]}"
    module_cls = routine_dict["class"]
    routine_cls = getattr(importlib.import_module(module_path), module_cls)

    # initialize the routine object
    routine = routine_cls(configs, verbose=routine_dict["verbose"])
    routine.apply()


if __name__ == "__main__":
    main()
