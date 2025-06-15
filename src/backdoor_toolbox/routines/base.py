import importlib
from abc import ABC, abstractmethod


class BaseRoutine(ABC):
    """
    Abstract base class for defining routines that perform a sequence of operations
    such as data preparation, model setup, and the application of attacks or defenses.

    Subclasses must implement the `apply()` method, which is called automatically.

    Methods:
        apply(): Execute the routine.
        _import_package(package): Dynamically import a Python package.
    """

    @abstractmethod
    def apply(self):
        """
        Apply the complete routine.

        This method must be implemented by all subclasses.
        It is automatically invoked during the execution flow.
        """
        pass

    def _import_package(self, package: str) -> object:
        """
        Dynamically import a package by name.

        Parameters:
            package (str): The name of the package to import.

        Returns:
            types.ModuleType: The imported Python module.

        Raises:
            ImportError: If the package cannot be found or imported.
        """
        try:
            module = importlib.import_module(package)
            return module
        except ModuleNotFoundError as e:
            raise ImportError(f"Error importing package '{package}': {e}")
