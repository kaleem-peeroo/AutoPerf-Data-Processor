import logging

from typing import List

from experiment_file import ExperimentFile

lg = logging.getLogger(__name__)

class ExperimentRun:
    def __init__(
        self, 
        s_exp_name: str = "", 
        s_run_name: str = "", 
        ls_csv_paths = List[str],
    ):
        if s_exp_name == "":
            raise ValueError("Experiment name must not be empty")

        if s_run_name == "":
            raise ValueError("Run name must not be empty")

        if len(ls_csv_paths) == 0:
            raise ValueError("Experiment csv paths must not be empty")

        self.s_exp_name = s_exp_name
        self.s_run_name = s_run_name
        self.lo_exp_files = [
            ExperimentFile(s_exp_name, s_csv_path) for s_csv_path in ls_csv_paths
        ]

    def has_good_data(self):
        """
        Check if the run has good data.
        """
        for o_exp_file in self.lo_exp_files:
            if not o_exp_file.is_valid():
                return False

        return True

    def has_raw_data(self):
        """
        Check if the run has raw data.
        """
        for o_exp_file in self.lo_exp_files:
            if not o_exp_file.is_raw():
                lg.warning(
                    f"{o_exp_file.s_exp_name} is not raw"
                )
                return False

        return True

    def get_total_sample_count(self):
        if len(self.lo_exp_files) == 0:
            raise ValueError("Experiment files must not be empty")

        total_sample_count = 0

        for o_exp_file in self.lo_exp_files:
            total_sample_count += o_exp_file.get_total_sample_count()

        return total_sample_count
