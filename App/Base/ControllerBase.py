import logging
import importlib

from App.Configs.Configs import PACKAGE_PATH
from App.Utils.utils import get_all_packages


class ControllerBase():
    def __init__(self, *args, **kwargs):
        self.PACKAGE_PATH = PACKAGE_PATH

    def _import_module(self, func_name):
        """
        Import some function from packages in self.PACKAGE_PATH
        :param func_name: __name__ of target function
            input "" -> return None
            if Module No Found -> exit!
        :return: target function
        """
        if func_name == "":
            return None
        l_py_files = get_all_packages(self.PACKAGE_PATH)
        result = []
        for py_file_i in l_py_files:
            try:
                pck_i = importlib.import_module(py_file_i)
                func_i = getattr(pck_i, func_name)
                result.append(func_i)
            except Exception as err:
                pass
        if len(result) == 0:
            logging.error(f"IMPORT MODULE : No module Named {func_name} is found in {self.PACKAGE_PATH}")
            exit(1)
        elif len(result) >= 2:
            logging.info(f"IMPORT MODULE : More than one {func_name} is found in {self.PACKAGE_PATH} -> {result}")
        return result[0]
