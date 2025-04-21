import os
import re

class ExperimentFile:
    def __init__(self, s_path: str = ""):
        if s_path == "":
            raise ValueError(
                "Experiment file path must not be empty"
            )

        if not os.path.exists(s_path):
            raise ValueError(
                f"Experiment file path does not exist: {s_path}"
            )

        if not s_path.endswith(".csv"):
            raise ValueError(
                f"Experiment file path must be a .csv file: {s_path}"
            )

        self.s_path = s_path

    def get_s_path(self):
        if self.s_path == "":
            raise ValueError(
                "Experiment file path must not be empty"
            )

        if not os.path.exists(self.s_path):
            raise ValueError(
                f"Experiment file path does not exist: {self.s_path}"
            )

        if not self.s_path.endswith(".csv"):
            raise ValueError(
                f"Experiment file path must be a .csv file: {self.s_path}"
            )

        return self.s_path

    def is_raw(self):
        s_path = self.get_s_path()
        s_file_name = os.path.basename(s_path)

        # Check if file contains *pub_0.csv or *sub_n.csv

        if re.search(r"pub_0\.csv|sub_\d+\.csv", s_file_name):
            return True

        else:
            return False
