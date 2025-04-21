import os
import logging
import pandas as pd

from rich.pretty import pprint
from typing import List

from experiment_run import ExperimentRun

lg = logging.getLogger(__name__)

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
            lg.info(
                f"{self.s_name} summary already exists. Skipping."
            )
            return

        if self.best_exp_run is None:    
            raise ValueError("No best experiment run found")

        df_summary = pd.DataFrame()
        for o_file in self.best_exp_run.lo_exp_files:
            
            if not o_file.is_raw():
                df = o_file.get_df()
                df_summary = pd.concat([df_summary, df], axis=0)

            elif o_file.is_pub():
                df_lat = self.get_lat_df(o_file)
                df_summary = pd.concat([df_summary, df_lat], axis=1)

            elif o_file.is_sub():
                df_mbps = self.get_mbps_df(o_file)
                df_summary = pd.concat([df_summary, df_mbps], axis=1)

            else:
                raise ValueError("Unknown file type")

        df_summary = self.calculate_sub_metrics(df_summary)

        df_summary['experiment_name'] = self.s_name

        df_summary.reset_index(drop=True, inplace=True)
        df_summary.to_parquet(s_output_path, index=False)
        lg.info(
            f"Summary file written to {s_output_path}"
        )

        return df_summary

    def get_lat_df(self, o_file):
        """
        Get latency dataframe.
        """
        if not o_file.is_pub():
            raise ValueError("File is not a publisher file")

        df = o_file.get_df()

        ls_lat_cols = [col for col in df.columns if "latency" in col.lower()]

        if len(ls_lat_cols) == 0:
            raise ValueError("No latency columns found in file")
        if len(ls_lat_cols) > 1:
            raise ValueError("Multiple latency columns found in file")

        s_lat_col = ls_lat_cols[0]

        sr = df[s_lat_col]
        sr = sr.dropna()
        sr.rename("latency_us", inplace=True)

        return sr

    def get_mbps_df(self, o_file):
        """
        Get mbps dataframe.
        """
        if not o_file.is_sub():
            raise ValueError("File is not a subscriber file")

        df = o_file.get_df()
        ls_mbps_cols = [
            col for col in df.columns if "mbps" in col.lower() \
                    and 'avg' not in col.lower()
        ]
        if len(ls_mbps_cols) == 0:
            raise ValueError("No mbps columns found in file")
        if len(ls_mbps_cols) > 1:
            raise ValueError("Multiple mbps columns found in file")

        s_mbps_col = ls_mbps_cols[0]

        sr = df[s_mbps_col]
        sr = sr.dropna()

        s_sub_name = os.path.basename(o_file.s_path)
        s_sub_name = s_sub_name.split(".")[0]

        # Rename and prepend with sub_n
        sr.rename(
            f"{s_sub_name}_mbps",
            inplace=True
        )
        
        return sr

    def calculate_sub_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate subscriber metrics.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "Input is not a dataframe"
                f"It is a {type(df)}"
            )

        if df.empty:
            raise ValueError("Dataframe is empty")

        # Calculate avg mbps per sub
        ls_mbps_cols = [col for col in df.columns if "mbps" in col.lower()]
        if len(ls_mbps_cols) == 0:
            raise ValueError("No mbps columns found in dataframe")

        df["avg_mbps_per_sub"] = df[ls_mbps_cols].mean(axis=1)
        df["total_mbps_over_subs"] = df[ls_mbps_cols].sum(axis=1)

        return df
