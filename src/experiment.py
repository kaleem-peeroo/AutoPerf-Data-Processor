import os
import logging
import pandas as pd

from rich.pretty import pprint
from typing import List

logger = logging.getLogger(__name__)

class Experiment:
    def __init__(
        self, 
        s_name: str = "", 
        ls_csv_paths = List[str],
    ):
        if s_name == "":
            raise ValueError("Experiment name must not be empty")

        if len(ls_csv_paths) == 0:
            raise ValueError("Experiment csv paths must not be empty")

        self.s_name = s_name
        self.ls_csv_paths = ls_csv_paths
        
    def __str__(self):
        return "Experiment: {}, CSV Paths: {}".format(
            self.s_name,
            [os.path.basename(s_csv_path) for s_csv_path in self.ls_csv_paths]
        )

    def get_name(self):
        if self.s_name == "":
            raise ValueError("Experiment name must not be empty")

        return self.s_name

    def add_csv_path(self, s_csv_path: str = ""):
        if s_csv_path == "":
            raise ValueError("CSV path must not be empty")

        if not os.path.exists(s_csv_path):
            raise ValueError("CSV path does not exist: {}".format(s_csv_path))

        if not s_csv_path.endswith(".csv"):
            raise ValueError("CSV path must be a .csv file: {}".format(s_csv_path))

        if s_csv_path not in self.ls_csv_paths:
            self.ls_csv_paths.append(s_csv_path)
