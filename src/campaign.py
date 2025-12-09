import os
import logging
import pandas as pd

from rich.pretty import pprint
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from experiment import Experiment

import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

lg = logging.getLogger(__name__)


class Campaign:
    def __init__(self, d_config):
        self._s_raw_dpath = d_config["exp_folders"]
        self._s_ap_conf_path = d_config["ap_config"]
        self._s_ds_output_path = d_config["dataset_path"]
        self._s_summaries_dpath = os.path.join(
            os.path.dirname(self._s_ds_output_path),
            f"{os.path.basename(self._s_ds_output_path).split('.')[0]}_summaries",
        )
        self._df_ds = None

    def summarise_experiments(self):
        """
        Goes through each experiment.
        Gathers all the data and puts it into a single dataframe.
        Writes the df to a parquet file.
        Stores the parquet file summaries_dpath.
        """
        lo_exps = self.gather_experiments(self._s_raw_dpath)

        os.makedirs(self._s_summaries_dpath, exist_ok=True)

        for i_exp, o_exp in enumerate(lo_exps):
            lg.info(
                f"[{i_exp + 1:,.0f}/{len(lo_exps):,.0f}]"
                f"Processing \n\t{o_exp.s_name}..."
            )
            o_exp.summarise(s_dpath=self._s_summaries_dpath)

    def create_dataset(self):
        """
        Read all experiment summaries and stick into one big dataset.
        """

        if not os.path.exists(self._s_summaries_dpath):
            raise Exception(
                f"Summaries directory does not exist: {self._s_summaries_dpath}. "
                f"You need to run summarise_experiments() first."
            )

        lg.debug("Creating dataset...")

        if os.path.exists(self._s_ds_output_path):
            lg.warning("Dataset already exists. Renaming with timestamp...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_ds_output = self._s_ds_output_path.replace(
                ".parquet", f"_{timestamp}.parquet"
            )
            os.rename(self._s_ds_output_path, new_ds_output)
            lg.warning(f"Dataset renamed to {new_ds_output}")

        df_ds = pd.DataFrame()

        ls_summaries = os.listdir(self._s_summaries_dpath)
        ls_summaries = [
            os.path.join(self._s_summaries_dpath, s_summary)
            for s_summary in ls_summaries
        ]
        ls_summaries = [
            s_summary for s_summary in ls_summaries if s_summary.endswith(".parquet")
        ]

        lg.info(f"Found {len(ls_summaries)} summaries in {self._s_summaries_dpath}")

        df_ds = pd.concat(
            (pd.read_parquet(s_summary) for s_summary in ls_summaries),
            ignore_index=True,
        )

        self._df_ds = df_ds
        self.write_dataset(df_ds)

    def write_dataset(self, df: pd.DataFrame = pd.DataFrame()) -> None:
        if df.empty:
            lg.warning("Input dataframe is empty")

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a dataframe: {df}")

        os.makedirs(os.path.dirname(self._s_ds_output_path), exist_ok=True)

        ls_num_cols = ["latency_us", "avg_mbps_per_sub", "total_mbps_over_subs"]
        for s_num_col in ls_num_cols:
            df[s_num_col] = pd.to_numeric(df[s_num_col], errors="coerce")

        df.to_parquet(self._s_ds_output_path, index=False)
        lg.info(f"Dataset written to {self._s_ds_output_path}")

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

    def gather_experiments(self, s_raw_datadir: str = "") -> List[Experiment]:
        """
        1. Get list of csv paths.
        2. Group into experiments (use experiment_name).
        3. Create Experiment objects.
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

        return lo_exps

    def process_exp_runs(self, lo_exps: List[Experiment] = []) -> List[Experiment]:
        if len(lo_exps) == 0:
            raise ValueError("No experiments found")

        for i_exp, o_exp in enumerate(lo_exps):
            s_counter = f"[{i_exp + 1:,.0f}/{len(lo_exps):,.0f}]"

            lg.debug(f"{s_counter} " f"Processing runs for {o_exp.s_name}")

            if not isinstance(o_exp, Experiment):
                raise ValueError(f"Experiment is not an Experiment object: {o_exp}")

            o_exp.process_runs()

        return lo_exps

    def process_csv_paths_into_experiments(
        self, ls_csv_paths: List[str] = []
    ) -> List[Experiment]:
        if not isinstance(ls_csv_paths, list):
            raise ValueError(f"CSV paths must be a list: {ls_csv_paths}")

        if len(ls_csv_paths) == 0:
            raise ValueError("No csv paths provided")

        lo_exps = []
        for i_csv_path, s_csv_path in enumerate(ls_csv_paths):
            s_counter = f"[{i_csv_path + 1:,.0f}/{len(ls_csv_paths):,.0f}]"
            if i_csv_path % 1000 == 0:
                lg.debug(f"{s_counter} Processing path: {os.path.basename(s_csv_path)}")

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
                o_exp = Experiment(s_name=s_exp_name, ls_csv_paths=[s_csv_path])
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
            raise ValueError(f"No experiment name found in file path: {s_fpath}")

        s_exp_name = ""
        ls_parts = s_fpath.split("/")
        for s_part in ls_parts:
            if "" in s_part:
                s_part = s_part.split(".")[0]
            if self.is_exp_name_in_str(s_part):
                s_exp_name = s_part
                break

        if s_exp_name == "":
            raise ValueError(f"No experiment name found in file path: {s_fpath}")

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

        lg.debug(f"Getting all fpaths in {s_exp_entry}")
        ls_fpaths = list(Path(s_exp_entry).rglob("*.*"))
        ls_fpaths = [str(fpath) for fpath in ls_fpaths]
        lg.debug(f"Found {len(ls_fpaths)} paths.")

        return ls_fpaths

    def validate_dataset(self):
        """
        Checks for cases in dataset where mbps has more than 600 samples.
        """
        df = self._df_ds

        if df is None:
            raise Exception("No dataset created yet")

        if df.empty:
            raise ValueError("Dataset is empty")

        if len(df.columns) == 0:
            raise ValueError("Dataset has no columns")

        if "experiment_name" not in df.columns:
            raise ValueError("Dataset must have experiment_name column")

        ls_exp_names = df["experiment_name"].unique().tolist()
        if len(ls_exp_names) == 0:
            raise ValueError("Dataset has no experiment names")

        pprint(df.columns.tolist())

        return
        for i_exp, s_exp_name in ls_exp_names:
            s_counter = f"[{i_exp + 1:,.0f}/{len(ls_exp_names):,.0f}]"

            lg.debug(f"{s_counter} " f"Validating {s_exp_name}")

            df_exp = df[df["experiment_name"] == s_exp_name].copy()

            df_avg_mbps = df_exp["avg_mbps_per_sub"].copy()
            df_avg_mbps.dropna(inplace=True)

            if len(df_avg_mbps) > 600:
                lg.warning(f"Experiment {s_exp_name} has more than 600 samples. ")
                continue

            df_total_mbps = df_exp["total_mbps_over_subs"].copy()
            df_total_mbps.dropna(inplace=True)

            if len(df_total_mbps) > 600:
                lg.warning(f"Experiment {s_exp_name} has more than 600 samples. ")
                continue
