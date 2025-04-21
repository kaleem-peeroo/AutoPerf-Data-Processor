from typing import List

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
        self.ls_csv_paths = ls_csv_paths
