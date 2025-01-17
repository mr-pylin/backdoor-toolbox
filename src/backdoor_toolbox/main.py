import importlib

from backdoor_toolbox.routine import routine
from backdoor_toolbox.utils.logger import Logger


def main():
    # initialize the routine
    module_path = f"{routine["root"]}.{routine["file"]}"
    module_cls = routine["class"]
    routine_cls = getattr(importlib.import_module(module_path), module_cls)

    # initialize the routine object
    routine_obj = routine_cls()
    routine_obj.apply()


if __name__ == "__main__":
    main()
