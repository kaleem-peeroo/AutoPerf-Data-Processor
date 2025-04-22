import os
import gc
import toml
import itertools
import sys
import logging
import re
import pandas as pd

from rich.pretty import pprint
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from itertools import islice

from utils import get_qos_name, calculate_averages, get_df_from_csv, aggregate_across_cols
from experiment import Experiment

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

lg = logging.getLogger(__name__)

class Campaign:
    def __init__(self, d_config):
        self.raw_datadir = d_config['exp_folders']
        self.apconf_path = d_config['ap_config']
        self.ds_output_path = d_config['dataset_path']
        self.s_summaries_dpath = os.path.join(
            os.path.dirname(self.ds_output_path),
            f"{os.path.basename(self.ds_output_path).split('.')[0]}_summaries"
        )
        self.df_ds = None

    def get_raw_datadir(self):
        if not self.raw_datadir:
            raise Exception("No raw data directory provided")

        if not isinstance(self.raw_datadir, str):
            raise ValueError(f"Raw data directory must be a string: {self.raw_datadir}")

        if self.raw_datadir == "":
            raise ValueError("Raw data directory must not be empty")

        if "~" in self.raw_datadir:
            self.raw_datadir = os.path.expanduser(self.raw_datadir)

        return self.raw_datadir

    def get_dataset_path(self):
        if not self.ds_output_path:
            raise Exception("No dataset path provided")

        if not isinstance(self.ds_output_path, str):
            raise ValueError(f"Dataset path must be a string: {self.ds_output_path}")

        if self.ds_output_path == "":
            raise ValueError("Dataset path must not be empty")

        if "~" in self.ds_output_path:
            self.ds_output_path = os.path.expanduser(self.ds_output_path)

        return self.ds_output_path

    def get_df_ds(self):
        if self.df_ds is None:
            raise Exception("No dataset created yet")

        if not isinstance(self.df_ds, pd.DataFrame):
            raise ValueError(f"Dataset is not a dataframe: {self.df_ds}")

        if self.df_ds.empty:
            raise ValueError("Dataset is empty")

        return self.df_ds

    def summarise_experiments(self):
        """
        Goes through each experiment.
        Gathers all the data and puts it into a single dataframe.
        Writes the df to a parquet file.
        Stores the parquet file summaries_dpath.
        """
        s_raw_datadir = self.get_raw_datadir()
        lo_exps = self.gather_experiments(s_raw_datadir)

        os.makedirs(self.s_summaries_dpath, exist_ok=True)

        for i_exp, o_exp in enumerate(lo_exps):
            s_counter = f"[{i_exp + 1}/{len(lo_exps)}]"
            lg.info(
                f"{s_counter} "
                f"Processing experiment: {o_exp.s_name}"
            )
            try:
                o_exp.process(s_dpath=self.s_summaries_dpath)

            except Exception as e:
                lg.error(
                    f"{s_counter} "
                    f"Error processing experiment: {o_exp.s_name}"
                )
                lg.error(e)
                continue

    def create_dataset(self):
        """
        Read all experiment summaries and stick into one big dataset.
        """

        if not os.path.exists(self.s_summaries_dpath):
            raise Exception(
                f"Summaries directory does not exist: {self.s_summaries_dpath}. "
                f"You need to run summarise_experiments() first."
            )

        lg.debug("Creating dataset...")

        ds_output = self.get_dataset_path()
        if os.path.exists(ds_output):
            lg.warning("Dataset already exists. Renaming with timestamp...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_ds_output = ds_output.replace(".parquet", f"_{timestamp}.parquet")
            os.rename(ds_output, new_ds_output)
            lg.warning(f"Dataset renamed to {new_ds_output}")

        df_ds = pd.DataFrame()

        ls_summaries = os.listdir(self.s_summaries_dpath)
        lg.info(f"Found {len(ls_summaries)} summaries in {self.s_summaries_dpath}")

        for i_summary, s_summary in enumerate(ls_summaries):
            s_counter = f"[{i_summary + 1}/{len(ls_summaries)}]"
            lg.debug(
                f"{s_counter} "
                f"Adding summary: {s_summary}"
            )

            if not s_summary.endswith(".parquet"):
                lg.warning(f"Skipping non-parquet file: {s_summary}")
                continue

            s_summary_path = os.path.join(self.s_summaries_dpath, s_summary)

            df_temp = pd.read_parquet(s_summary_path)
            df_ds = pd.concat([df_ds, df_temp], axis=0)
            
        self.df_ds = df_ds

        self.write_dataset(df_ds)

    def write_dataset(
        self,
        df: pd.DataFrame = pd.DataFrame()
    ) -> None:
        if df.empty:
            lg.warning("Input dataframe is empty")

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a dataframe: {df}")

        s_ds_output = self.get_dataset_path()
        os.makedirs(os.path.dirname(s_ds_output), exist_ok=True)

        df.to_parquet(s_ds_output, index=False)
        lg.info(f"Dataset written to {self.ds_output_path}")

    def get_sub_mbps_cols(self, df: pd.DataFrame = pd.DataFrame()) -> List[str]:
        """
        Gets all sub_n_mbps columns from the dataframe.
        """
        if df.empty:
            raise ValueError("Dataframe is empty")

        if len(df.columns) == 0:
            raise ValueError("Dataframe has no columns")

        ls_sub_mbps_cols = []
        for col in df.columns:
            if "sub_" in col and "_mbps" in col:
                ls_sub_mbps_cols.append(col)

        return ls_sub_mbps_cols
    
    def add_input_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input dataframe is empty")

        if 'experiment_name' not in df.columns:
            raise ValueError("Input dataframe must have experiment_name column")

        if df['experiment_name'].nunique() != 1:
            raise ValueError("Input dataframe must have only one experiment name")

        s_exp_name = df['experiment_name'].iloc[0]
        if not self.follows_experiment_name_format(s_exp_name):
            s_exp_name = self.try_format_experiment_name(s_exp_name)
            if not self.follows_experiment_name_format(s_exp_name):
                raise ValueError(
                    f"Experiment name does not follow expected format: {s_exp_name}"
                )

        d_qos = self.get_qos_from_exp_name(s_exp_name)

        for key, value in d_qos.items():
            if key not in df.columns:
                df[key] = value
                df[key] = df[key].astype('float64')

        return df

    def get_qos_from_exp_name(
        self, 
        s_exp_name: str = ""
    ) -> Dict[str, str]:
        if s_exp_name == "":
            raise Exception("No experiment name provided")

        if "_" not in s_exp_name:
            raise ValueError(f"Experiment name must have underscores: {s_exp_name}")

        if not self.follows_experiment_name_format(s_exp_name):
            raise ValueError(
                f"Experiment name does not follow expected format: {s_exp_name}"
            )

        if not self.follows_experiment_name_format(s_exp_name):
            s_exp_name = self.try_format_experiment_name(s_exp_name)
            if not self.follows_experiment_name_format(s_exp_name):
                raise ValueError(
                    f"Experiment name does not follow expected format: {s_exp_name}"
                )

        ls_parts = s_exp_name.split("_")

        d_qos = {
            "duration_secs": ls_parts[0].split("SEC")[0],
            "datalen_bytes": ls_parts[1].split("B")[0],
            "pub_count": ls_parts[2].split("P")[0],
            "sub_count": ls_parts[3].split("S")[0],
            "use_reliable": 1 if "REL" in ls_parts[4] else 0,
            "use_multicast": 1 if "MC" in ls_parts[5] else 0,
            "durability": ls_parts[6].split("DUR")[0],
            "latency_count": ls_parts[7].split("LC")[0]
        }

        # Convert to int
        for key in d_qos.keys():
            # Use regex to remove non-numeric characters
            d_qos[key] = int(d_qos[key])

        return d_qos

    def gather_experiments(
        self, 
        s_raw_datadir: str = ""
    ) -> List[Experiment]:
        """
        1. Get list of csv paths.
        2. Group into experiments (use experiment_name).
        3. Create Experiment objects.
        4. Find best run per o_exp
        5. Store experiment objects in a list as self.lo_exps
        """

        lg.debug("Gathering experiments...")

        if s_raw_datadir == "":
            raise Exception("No raw data directory provided")

        if not os.path.exists(s_raw_datadir):
            raise Exception(f"Raw data directory does not exist: {s_raw_datadir}")

        if not os.path.isdir(s_raw_datadir):
            raise Exception(f"Raw data directory is not a directory: {s_raw_datadir}")

        lg.debug(f"Getting exps in {s_raw_datadir}...")

        ls_fpaths = self.recursively_get_fpaths(s_raw_datadir)
        ls_csv_paths = [_ for _ in ls_fpaths if _.endswith(".csv")]
        if len(ls_csv_paths) == 0:
            raise ValueError(f"No csv files found in {s_raw_datadir}")

        lg.debug(f"Found {len(ls_csv_paths)} csv files in {s_raw_datadir}")

        lo_exps = self.process_csv_paths_into_experiments(ls_csv_paths)

        lo_exps = self.process_exp_runs(lo_exps)

        lo_exps = self.pick_best_exp_run(lo_exps)

        return lo_exps

    def process_exp_runs(
        self,
        lo_exps: List[Experiment] = []
    ) -> List[Experiment]:
        if len(lo_exps) == 0:
            raise ValueError("No experiments found")

        for i_exp, o_exp in enumerate(lo_exps):
            s_counter = f"[{i_exp + 1:,.0f}/{len(lo_exps):,.0f}]"

            lg.debug(
                f"{s_counter} "
                f"Processing runs for {o_exp.s_name}"
            )

            if not isinstance(o_exp, Experiment):
                raise ValueError(f"Experiment is not an Experiment object: {o_exp}")

            try:
                o_exp.process_runs()
            except Exception as e:
                lg.error(
                    f"{s_counter} "
                    f"Error processing runs for {o_exp.s_name}"
                )
                lg.error(e)
                continue

        return lo_exps

    def pick_best_exp_run(
        self,
        lo_exps: List[Experiment] = []
    ) -> List[Experiment]:
        if len(lo_exps) == 0:
            raise ValueError("No experiments found")

        for i_exp, o_exp in enumerate(lo_exps):
            s_counter = f"[{i_exp + 1:,.0f}/{len(lo_exps):,.0f}]"

            lg.debug(
                f"{s_counter} "
                f"Picking best run for {o_exp.s_name}"
            )

            if not isinstance(o_exp, Experiment):
                raise ValueError(f"Experiment is not an Experiment object: {o_exp}")

            try:
                o_exp.pick_best_run()
            except Exception as e:
                lg.error(
                    f"{s_counter} "
                    f"Error picking best run for {o_exp.s_name}"
                )
                lg.error(e)
                continue

        return lo_exps

    def process_csv_paths_into_experiments(
        self,
        ls_csv_paths: List[str] = []
    ) -> List[Experiment]:
        if not isinstance(ls_csv_paths, list):
            raise ValueError(f"CSV paths must be a list: {ls_csv_paths}")

        if len(ls_csv_paths) == 0:
            raise ValueError("No csv paths provided")

        lo_exps = []
        for i_csv_path, s_csv_path in enumerate(ls_csv_paths):
            s_counter = f"[{i_csv_path + 1:,.0f}/{len(ls_csv_paths):,.0f}]"
            lg.debug(
                f"{s_counter} Processing path: {os.path.basename(s_csv_path)}"
            )

            if not os.path.isfile(s_csv_path):
                raise ValueError(f"CSV path is not a file: {s_csv_path}")

            if not s_csv_path.endswith(".csv"):
                raise ValueError(f"CSV path is not a csv file: {s_csv_path}")

            ls_exp_names = [o_exp.get_name() for o_exp in lo_exps]
            s_exp_name = self.get_experiment_name_from_fpath(s_csv_path)

            if s_exp_name in ls_exp_names:
                o_exp = lo_exps[ls_exp_names.index(s_exp_name)]
                o_exp.add_csv_path(s_csv_path)

            else:
                o_exp = Experiment(
                    s_name=s_exp_name,
                    ls_csv_paths=[ s_csv_path ]
                )
                lo_exps.append(o_exp)

        if len(lo_exps) == 0:
            raise ValueError("No experiments found")

        return lo_exps

    def get_experiment_name_from_fpath(self, s_fpath: str = ""):
        if s_fpath == "":
            raise Exception("No experiment entry provided")

        if not isinstance(s_fpath, str):
            raise ValueError(f"Experiment entry must be a string: {s_fpath}")

        if not self.is_experiment_name_in_fpath(s_fpath):
            raise ValueError(
                f"No experiment name found in file path: {s_fpath}"
            )

        s_exp_name = ""
        ls_parts = s_fpath.split("/")
        for s_part in ls_parts:
            if self.is_exp_name_in_str(s_part):
                s_exp_name = s_part
                break

        if s_exp_name == "":
            raise ValueError(
                f"No experiment name found in file path: {s_fpath}"
            )

        return s_exp_name
        
    def is_experiment_name_in_fpath(self, s_fpath: str = "") -> bool:
        """
        Checks if the experiment name is in the file path.
        1. Break fpath into parts.
        2. For each part check if it matches experiment name format.
        """
        if s_fpath == "":
            raise Exception("No experiment entry provided")

        if not isinstance(s_fpath, str):
            raise ValueError(f"Experiment entry must be a string: {s_fpath}")

        b_exp_in_dir = False

        ls_parts = s_fpath.split("/")
        for s_part in ls_parts:
            if self.is_exp_name_in_str(s_part):
                b_exp_in_dir = True
                break

        return b_exp_in_dir
        
    def is_exp_name_in_str(self, s_value: str = "") -> bool:
        """
        Checks if the experiment name is in the string.
        """
        if s_value == "":
            return False

        s_value = s_value.upper()

        i_underscore_count = s_value.count("_")
        if i_underscore_count == 0:
            return False

        if i_underscore_count != 7 and i_underscore_count != 8:
            return False

        if "." in s_value:
            s_value = s_value.split(".")[0]

        if s_value.endswith("LC"):
            return True

        return False
    def follows_experiment_name_format(self, s_filename: str = "") -> bool:
        """
        Checks if filename follows following format:
        *{int}SEC_{int}B_{int}P_{int}S_{REL/BE}_{MC/UC}_{int}DUR_{int}LC*
        """

        if s_filename == "":
            raise Exception("No filename provided")

        if "." in s_filename:
            s_filename = s_filename.split(".")[0]

        ls_parts = s_filename.upper().split("_")
        ld_end_matches = [
            {"values": ["SEC"]},
            {"values": ["B"]},
            {"values": ["PUB"]},
            {"values": ["SUB"]},
            {"values": ["REL", "BE"]},
            {"values": ["MC", "UC"]},
            {"values": ["DUR"]},
            {"values": ["LC"]},
        ]

        for i, part in enumerate(ls_parts):
            ls_end_matches = ld_end_matches[i]["values"]

            b_match = False
            for s_end_match in ls_end_matches:
                if part.endswith(s_end_match):
                    b_match = True
                    break

            if not b_match:
                return False
                                        
        return True
        
    def recursively_get_fpaths(self, s_exp_entry: str = "") -> List[str]:
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")

        if not os.path.exists(s_exp_entry):
            raise Exception(f"Experiment entry does not exist: {s_exp_entry}")

        if os.path.isfile(s_exp_entry):
            return [s_exp_entry]

        if not os.path.isdir(s_exp_entry):
            raise Exception(f"Experiment entry is not a directory: {s_exp_entry}")

        ls_fpaths = list(Path(s_exp_entry).rglob("*.*"))
        ls_fpaths = [str(fpath) for fpath in ls_fpaths]

        return ls_fpaths

    def validate_dataset(self):
        """
        Checks for cases in dataset where mbps has more than 600 samples.
        """
        df = pd.read_parquet(self.get_dataset_path())
        if df.empty:
            raise ValueError("Dataset is empty")

        if len(df.columns) == 0:
            raise ValueError("Dataset has no columns")

        if 'experiment_name' not in df.columns:
            raise ValueError("Dataset must have experiment_name column")

        ls_exp_names = df['experiment_name'].unique().tolist()
        if len(ls_exp_names) == 0:
            raise ValueError("Dataset has no experiment names")

        for s_exp_name in ls_exp_names:
            df_exp = df[df['experiment_name'] == s_exp_name].copy()

            df_avg_mbps = df_exp['avg_mbps_per_sub'].copy()
            df_avg_mbps.dropna(inplace=True)

            if len(df_avg_mbps) > 600:
                lg.warning(
                    f"Experiment {s_exp_name} has more than 600 samples. "
                )
                continue

            df_total_mbps = df_exp['total_mbps_over_subs'].copy()
            df_total_mbps.dropna(inplace=True)

            if len(df_total_mbps) > 600:
                lg.warning(
                    f"Experiment {s_exp_name} has more than 600 samples. "
                )
                continue
