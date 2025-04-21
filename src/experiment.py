import os
import logging
import pandas as pd

from rich.pretty import pprint
from typing import List

from experiment_run import ExperimentRun

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

        self.lo_exp_runs = []
        self.best_exp_run = None
        
    def __str__(self):
        return "Experiment: {}, CSV Paths: {}".format(
            self.s_name,
            [os.path.basename(s_csv_path) for s_csv_path in self.ls_csv_paths]
        )

    def __repr__(self):
        return "Experiment: {}, CSV Paths: {}".format(
            self.s_name,
            [os.path.basename(s_csv_path) for s_csv_path in self.ls_csv_paths]
        )

    def get_name(self):
        if self.s_name == "":
            raise ValueError("Experiment name must not be empty")

        return self.s_name

    def get_csv_paths(self) -> List[str]:
        if len(self.ls_csv_paths) == 0:
            raise ValueError("Experiment csv paths must not be empty")

        return self.ls_csv_paths

    def add_csv_path(self, s_csv_path: str = ""):
        if s_csv_path == "":
            raise ValueError("CSV path must not be empty")

        if not os.path.exists(s_csv_path):
            raise ValueError("CSV path does not exist: {}".format(s_csv_path))

        if not s_csv_path.endswith(".csv"):
            raise ValueError("CSV path must be a .csv file: {}".format(s_csv_path))

        if s_csv_path not in self.ls_csv_paths:
            self.ls_csv_paths.append(s_csv_path)

    def process_runs(self):
        """
        Differentiate between csv paths and run names.
        """
        ls_run_names = self.get_run_names()
        ls_csv_paths = self.get_csv_paths()
        
        for s_run in ls_run_names:
            ls_run_csvs = [_ for _ in ls_csv_paths if s_run in _]

            o_exp_run = ExperimentRun(
                s_exp_name=self.s_name,
                s_run_name=s_run,
                ls_csv_paths=ls_run_csvs
            )

            self.lo_exp_runs.append(o_exp_run)

    def get_run_names(self) -> List[str]:
        """
        Get unique run names from the CSV paths.
        """
        ls_run_names = []

        for s_csv_path in self.ls_csv_paths:
            s_fname = os.path.basename(s_csv_path)
            s_run_name = os.path.dirname(s_csv_path)

            if s_run_name not in ls_run_names:
                ls_run_names.append(s_run_name)

        return ls_run_names

    def pick_best_run(self):
        if len(self.lo_exp_runs) == 0:
            raise ValueError("No experiment runs found")

        lo_good_runs = self.get_good_exp_runs()
        if len(lo_good_runs) > 0:
            self.best_exp_run = lo_good_runs[0]

        # If no good runs, pick the one with raw files
        lo_raw_runs = self.get_raw_exp_runs()
        if len(lo_raw_runs) > 0:
            self.best_exp_run = lo_raw_runs[0]

        # If no raw runs, pick the first one
        self.best_exp_run = self.lo_exp_runs[0]

    def get_good_exp_runs(self) -> List[ExperimentRun]:
        """
        Get all experiment runs that have good data.
        """
        lo_good_runs = []

        for o_exp_run in self.lo_exp_runs:
            if o_exp_run.has_good_data():
                lo_good_runs.append(o_exp_run)

        # Sort by total sample count decreasing
        lo_good_runs = self.sort_by_total_sample_count(lo_good_runs)
        return lo_good_runs

    def sort_by_total_sample_count(
        self, 
        lo_exp_runs: List[ExperimentRun]
    ) -> List[ExperimentRun]:
        """
        Sort experiment runs by total sample count.
        """
        lo_exp_runs.sort(
            key=lambda x: x.get_total_sample_count(), 
            reverse=True
        )
        return lo_exp_runs

    def get_raw_exp_runs(self) -> List[ExperimentRun]:
        """
        Get all experiment runs that have raw data.
        """
        lo_raw_runs = []

        for o_exp_run in self.lo_exp_runs:
            if o_exp_run.has_raw_data():
                lo_raw_runs.append(o_exp_run)

        # Sort by total sample count decreasing
        lo_raw_runs = self.sort_by_total_sample_count(lo_raw_runs)

        return lo_raw_runs
        
    def process(self, s_dpath: str = ""):
        """
        1. Summarise.
        2. Write summary file to s_dpath as {exp_name}.parquet.
        """

        if s_dpath == "":
            raise ValueError("Output path must not be empty")

        os.makedirs(s_dpath, exist_ok=True)
        s_output_path = os.path.join(
            s_dpath,
            f"{self.s_name}.parquet"
        )

        if os.path.exists(s_output_path):
            logger.info(
                f"{self.s_name} summary already exists. Skipping."
            )
            return

        
        
