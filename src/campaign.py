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

# from logger import logger
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
        ld_exp_names_and_paths = self.get_experiments(s_raw_datadir)

        os.makedirs(self.s_summaries_dpath, exist_ok=True)

        for i_exp, d_exp_names_and_paths in enumerate(ld_exp_names_and_paths):
            s_exp_name = d_exp_names_and_paths['name']
            s_exp_summ_path = os.path.join(
                self.s_summaries_dpath,
                f"{s_exp_name}.parquet"
            )

            if os.path.exists(s_exp_summ_path):
                lg.info(
                    f"{s_exp_summ_path} summary exists. Skipping..."
                )
                continue

            try:
                df_exp = self.process_exp_df(d_exp_names_and_paths)

            except Exception as e:
                lg.error(e)
                continue

            if df_exp.empty:
                lg.warning(f"Experiment dataframe is empty: {d_exp_names_and_paths['name']}")
                continue

            df_exp.reset_index(drop=True, inplace=True)
            df_exp.to_parquet(
                s_exp_summ_path,
                index=False
            )

            lg.info(
                f"[{i_exp + 1}/{len(ld_exp_names_and_paths)}] "
                f"Written {s_exp_name} summary to {self.s_summaries_dpath}"
            )

            del df_exp
            gc.collect()

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

        for i_summ, s_summ_path in enumerate(ls_summaries):
            s_summ_path = os.path.join(self.s_summaries_dpath, s_summ_path)

            lg.info(
                f"[{i_summ + 1}/{len(ls_summaries)}] "
                f"Processing {s_summ_path}"
            )

            if not os.path.isfile(s_summ_path):
                lg.warning(f"{s_summ_path} is not a file. Skipping...")
                continue

            if not s_summ_path.endswith(".parquet"):
                lg.warning(f"{s_summ_path} is not a parquet file. Skipping...")
                continue

            try:
                df_temp = pd.read_parquet(s_summ_path)
                df_ds = pd.concat([df_ds, df_temp], axis=0)

            except Exception as e:
                lg.error(e)
                continue
        
        if df_ds.empty:
            raise Exception("No data found in the dataset")

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

    def process_exp_df(
        self, 
        d_exp_names_and_paths: Dict[str, List[str]] = {}
    ) -> pd.DataFrame:
        """
        Generates a single df for the experiment.
        The df should have the following columns:
            - experiment_name
            - latency_us
            - avg_mbps_per_sub
            - total_mbps_over_subs
            - all input cols
        """
        if not d_exp_names_and_paths:
            raise Exception("No experiment names and paths provided")

        if 'name' not in d_exp_names_and_paths:
            raise ValueError("Experiment names and paths must have a name key")

        if 'paths' not in d_exp_names_and_paths:
            raise ValueError("Experiment names and paths must have a paths key")

        s_exp_name = d_exp_names_and_paths['name']
        ls_exp_paths = d_exp_names_and_paths['paths']

        df_exp = pd.DataFrame()

        for i_exp_path, s_exp_path in enumerate(ls_exp_paths):
            df_temp = self.get_exp_file_df(s_exp_path)
            df_exp = pd.concat([df_exp, df_temp], axis=1)

        df_exp.reset_index(drop=True, inplace=True)

        if df_exp.empty:
            raise ValueError(f"Experiment dataframe is empty: {s_exp_name}")

        if len(df_exp.columns) == 0:
            raise ValueError(f"Experiment dataframe has no columns: {s_exp_name}")

        if 'experiment_name' in df_exp.columns:
            df_exp.drop(columns=['experiment_name'], inplace=True)

        df_exp['experiment_name'] = self.try_format_experiment_name(s_exp_name)

        df_exp = self.add_input_cols(df_exp)

        df_exp = self.calculate_avg_mbps_per_sub(df_exp)
        df_exp = self.calculate_total_mbps_over_subs(df_exp)

        return df_exp

    def calculate_avg_mbps_per_sub(
        self,
        df: pd.DataFrame = pd.DataFrame()
    ) -> pd.DataFrame:
        """
        Gets all sub_n_mbps columns and calculates the average horizontally.
        Adds a new column called avg_mbps_per_sub.
        """
        if df.empty:
            raise ValueError("Dataframe is empty")

        if len(df.columns) == 0:
            raise ValueError("Dataframe has no columns")

        if "avg_mbps_per_sub" in df.columns:
            df.drop(columns=["avg_mbps_per_sub"], inplace=True)

        ls_sub_mbps_cols = self.get_sub_mbps_cols(df)
        if len(ls_sub_mbps_cols) == 0:
            raise ValueError("No sub_n_mbps columns found in dataframe")

        df_sub_mbps = df[ls_sub_mbps_cols]
        df_avg_mbps = df_sub_mbps.mean(axis=1)
        df_avg_mbps.name = "avg_mbps_per_sub"
        df = pd.concat([df, df_avg_mbps], axis=1)

        return df

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
    
    def calculate_total_mbps_over_subs(
        self,
        df: pd.DataFrame = pd.DataFrame()
    ) -> pd.DataFrame:
        """
        Gets all sub_n_mbps columns and calculates the total horizontally.
        Adds a new column called total_mbps_over_subs.
        """
        if df.empty:
            raise ValueError("Dataframe is empty")

        if len(df.columns) == 0:
            raise ValueError("Dataframe has no columns")

        if "total_mbps_over_subs" in df.columns:
            df.drop(columns=["total_mbps_over_subs"], inplace=True)

        ls_sub_mbps_cols = self.get_sub_mbps_cols(df)
        if len(ls_sub_mbps_cols) == 0:
            raise ValueError("No sub_n_mbps columns found in dataframe")

        df_sub_mbps = df[ls_sub_mbps_cols]
        df_total_mbps = df_sub_mbps.sum(axis=1)
        df_total_mbps.name = "total_mbps_over_subs"
        df = pd.concat([df, df_total_mbps], axis=1)
        
        return df

    def get_exp_file_df(
        self,
        s_exp_path: str = ""
    ) -> pd.DataFrame:
        """
        Get the experiment file dataframe.
        If its a raw file, process it.
        Otherwise, just read the csv file.
        """
        if os.path.getsize(s_exp_path) == 0:
            raise ValueError(f"Experiment file is empty: {s_exp_path}")

        if self.is_raw_exp_file(s_exp_path):
            if self.raw_file_is_pub(s_exp_path):
                df_temp = self.process_pub_file_df(s_exp_path)

            else:
                df_temp = self.process_sub_file_df(s_exp_path)

        else:
            df_temp = pd.read_csv(s_exp_path)

        return df_temp

    def is_raw_exp_file(
        self,
        s_exp_path: str = ""
    ) -> bool:
        """
        Checks if the file is raw (pub_0.csv or sub_n.csv) or processed.
        """
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        if not isinstance(s_exp_path, str):
            raise ValueError(f"Experiment path must be a string: {s_exp_path}")

        s_filename = os.path.basename(s_exp_path)

        if s_filename.startswith("pub_0") or re.match(r"^sub_\d+\.csv$", s_filename):
            return True

        else:
            if not self.follows_experiment_name_format(s_filename):
                s_filename = self.try_format_experiment_name(s_filename)
                if not self.follows_experiment_name_format(s_filename):
                    return False

        return False

    def process_pub_file_df(
        self,
        s_exp_path: str = ""
    ) -> pd.DataFrame:
        """
        Read the file manually and get the necessary columns.
        """
        
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        if not os.path.exists(s_exp_path):
            raise Exception(f"Experiment path does not exist: {s_exp_path}")

        if not os.path.isfile(s_exp_path):
            raise Exception(f"Experiment path is not a file: {s_exp_path}")

        if not s_exp_path.endswith(".csv"):
            raise Exception(f"Experiment path is not a csv file: {s_exp_path}")

        i_start = self.get_start_index_for_pub_file(s_exp_path)
        i_end = self.get_end_index_for_pub_file(s_exp_path)

        df = pd.read_csv(
            s_exp_path,
            skiprows=i_start,
            nrows=i_end - i_start,
            on_bad_lines="skip",
        )

        s_metric_col = self.get_metric_col_from_df(df, "latency")
        df = df[[s_metric_col]]

        df = self.rename_df_col(
            df=df,
            s_old_colname=s_metric_col,
            s_new_colname="latency_us",
        )

        for col in df.columns:
            try:
                df[col] = df[col].astype('float64')
            except ValueError:
                lg.error(f"Could not convert column {col} to float64: {df[col]}")
                    
        return df

    def process_sub_file_df(
        self,
        s_exp_path: str = ""
    ) -> pd.DataFrame:
        """
        Read the file manually and get the necessary columns.
        """
        
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        if not os.path.exists(s_exp_path):
            raise Exception(f"Experiment path does not exist: {s_exp_path}")

        if not os.path.isfile(s_exp_path):
            raise Exception(f"Experiment path is not a file: {s_exp_path}")

        if not s_exp_path.endswith(".csv"):
            raise Exception(f"Experiment path is not a csv file: {s_exp_path}")

        i_start = self.get_start_index_for_sub_file(s_exp_path)
        i_end = self.get_end_index_for_sub_file(s_exp_path)

        df = pd.read_csv(
            s_exp_path,
            skiprows=i_start,
            nrows=i_end - i_start,
            on_bad_lines="skip",
        )

        s_metric_col = self.get_metric_col_from_df(df, "mbps")
        df = df[[s_metric_col]]

        s_metric = f"{os.path.basename(s_exp_path).split(".")[0]}_mbps"

        df = self.rename_df_col(
            df=df,
            s_old_colname=s_metric_col,
            s_new_colname=s_metric,
        )

        for col in df.columns:
            try:
                df[col] = df[col].astype('float64')
            except ValueError:
                lg.error(f"Could not convert column {col} to float64: {df[col]}")
                    
        return df

    def rename_df_col(
        self,
        df: pd.DataFrame = pd.DataFrame(),
        s_old_colname: str = "",
        s_new_colname: str = ""
    ) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Dataframe is empty")

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a dataframe: {df}")

        if s_old_colname == "":
            raise ValueError("No old column name provided")

        if s_new_colname == "":
            raise ValueError("No new column name provided")

        if not isinstance(s_old_colname, str):
            raise ValueError(f"Old column name must be a string: {s_old_colname}")

        if not isinstance(s_new_colname, str):
            raise ValueError(f"New column name must be a string: {s_new_colname}")

        if s_old_colname not in df.columns:
            raise ValueError(f"Old column name not found in dataframe: {s_old_colname}")

        df.rename(columns={s_old_colname: s_new_colname}, inplace=True)

        return df

    def get_metric_col_from_df(
        self,
        df: pd.DataFrame = pd.DataFrame(),
        s_metric: str = ""
    ) -> str:
        if df.empty:
            raise ValueError("Dataframe is empty")

        if s_metric == "":
            raise ValueError("No metric provided")

        ls_cols = df.columns.tolist()
        s_colname = ""
        for s_col in ls_cols:
            if s_metric in s_col.lower() and "avg" not in s_col.lower():
                s_colname = s_col
                break

        if s_colname == "":
            raise ValueError(f"Could not find metric column in df: {s_metric}, {ls_cols}")

        if s_colname not in df.columns:
            raise ValueError(f"Metric column not found in dataframe: {s_colname}")

        return s_colname
        
    def raw_file_is_pub(
        self,
        s_exp_path: str = ""
    ) -> bool:
        """
        Checks if the raw file is a pub or sub file.
        """
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        s_filename = os.path.basename(s_exp_path)
        if s_filename.startswith("pub_0"):
            return True

        elif re.match(r"^sub_\d+\.csv$", s_filename):
            return False

        else:
            raise ValueError(
                f"Experiment path does not follow expected format: {s_exp_path}"
            )

    def get_start_index_for_pub_file(
        self,
        s_exp_path: str = ""
    ) -> int:
        """
        Looks for where the column titles start and returns the row index.
        """
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        if not os.path.exists(s_exp_path):
            raise Exception(f"Experiment path does not exist: {s_exp_path}")

        if not os.path.isfile(s_exp_path):
            raise Exception(f"Experiment path is not a file: {s_exp_path}")

        if not s_exp_path.endswith(".csv"):
            raise Exception(f"Experiment path is not a csv file: {s_exp_path}")

        with open(s_exp_path, "r") as o_file:
            ls_first_5_lines = []
            for i in range(5):
                line = o_file.readline()
                if not line:
                    break
                ls_first_5_lines.append(line)

        start_index = 0
        for i, line in enumerate(ls_first_5_lines):
            if "Length (Bytes)" in line:
                start_index = i
                break

        if start_index == 0:
            raise ValueError(f"Could not find start index for raw file: {s_exp_path}")

        return start_index

    def get_start_index_for_sub_file(
        self,
        s_exp_path: str = ""
    ) -> int:
        """
        Read the file N lines at a time and look for the last instance of "interval"
        """
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        if not os.path.exists(s_exp_path):
            raise Exception(f"Experiment path does not exist: {s_exp_path}")

        if not os.path.isfile(s_exp_path):
            raise Exception(f"Experiment path is not a file: {s_exp_path}")

        if not s_exp_path.endswith(".csv"):
            raise Exception(f"Experiment path is not a csv file: {s_exp_path}")

        i_file_line_count = sum(1 for _ in open(s_exp_path))
        i_chunk_size = i_file_line_count // 10

        li_interval_lines = []
        with open(s_exp_path, "rb") as o_file:
            for i_chunk, chunk in enumerate(iter(
                lambda: tuple(islice(o_file, i_chunk_size)), ()
            )):
                for i_line, s_line in enumerate(chunk):
                    i_line_count = i_chunk * i_chunk_size + i_line

                    if b"interval" in s_line.lower():
                        li_interval_lines.append(i_line_count)

        if len(li_interval_lines) == 0:
            raise ValueError(
                f"Could not find start index for raw file: {s_exp_path}"
            )

        i_last_interval = li_interval_lines[-1]

        return i_last_interval + 1

    def get_end_index_for_pub_file(
        self,
        s_exp_path: str = ""
    ) -> int:
        """
        Read the file N lines at a time and look for the summary line.
        """
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        if not os.path.exists(s_exp_path):
            raise Exception(f"Experiment path does not exist: {s_exp_path}")

        if not os.path.isfile(s_exp_path):
            raise Exception(f"Experiment path is not a file: {s_exp_path}")

        if not s_exp_path.endswith(".csv"):
            raise Exception(f"Experiment path is not a csv file: {s_exp_path}")

        i_file_line_count = sum(1 for _ in open(s_exp_path))
        i_chunk_size = i_file_line_count // 10

        b_found = False
        end_index = 0
        with open(s_exp_path, "rb") as o_file:
            for i_chunk, chunk in enumerate(iter(
                lambda: tuple(islice(o_file, i_chunk_size)), ()
            )):
                if b_found:
                    break

                for i_line, s_line in enumerate(chunk):
                    i_line_count = i_chunk * i_chunk_size + i_line
                    if b"summary" in s_line.lower() and not b_found:
                        end_index = i_line_count
                        b_found = True
                        break

                    elif b"interval" in s_line.lower() and \
                            not b_found and \
                            i_line_count > 10:
                        end_index = i_line_count
                        b_found = True
                        break

        if end_index <= 0 and not b_found:
            raise ValueError(
                f"Could not find end index for raw file: {s_exp_path}"
            )

        return end_index - 2

    def get_end_index_for_sub_file(
        self,
        s_exp_path: str = ""
    ) -> int:
        """
        Read the file N lines at a time and look for the summary line.
        """
        if s_exp_path == "":
            raise Exception("No experiment path provided")

        if not os.path.exists(s_exp_path):
            raise Exception(f"Experiment path does not exist: {s_exp_path}")

        if not os.path.isfile(s_exp_path):
            raise Exception(f"Experiment path is not a file: {s_exp_path}")

        if not s_exp_path.endswith(".csv"):
            raise Exception(f"Experiment path is not a csv file: {s_exp_path}")

        i_file_line_count = sum(1 for _ in open(s_exp_path))
        i_chunk_size = i_file_line_count // 10

        b_found = False
        end_index = 0
        with open(s_exp_path, "rb") as o_file:
            for i_chunk, chunk in enumerate(iter(
                lambda: tuple(islice(o_file, i_chunk_size)), ()
            )):
                if b_found:
                    break

                for i_line, s_line in enumerate(chunk):
                    i_line_count = i_chunk * i_chunk_size + i_line
                    if b"summary" in s_line.lower() and not b_found:
                        end_index = i_line_count
                        b_found = True
                        break

        if end_index <= 0 and not b_found:
            lg.warning(f"Couldn't find 'summary' in {s_exp_path}. Using last line.")
            end_index = i_file_line_count
            
        return end_index - 2
        
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

    def get_experiments(
        self, 
        s_raw_datadir: str = ""
    ) -> List[Dict[str, List[str]]]:
        """
        Gather a list of experiments.
        In the format of {name: str, paths: list}.
        Name is experiment name.
        Paths is a list of all files for that experiment.
            This could be a single file or multiple files.
        """
        lg.debug("Gathering experiments...")

        if s_raw_datadir == "":
            raise Exception("No raw data directory provided")

        if not os.path.exists(s_raw_datadir):
            raise Exception(f"Raw data directory does not exist: {s_raw_datadir}")

        if not os.path.isdir(s_raw_datadir):
            raise Exception(f"Raw data directory is not a directory: {s_raw_datadir}")

        lg.debug(f"Getting exps in {s_raw_datadir}...")

        ld_exp_names_and_paths = []
        ls_exp_entries = os.listdir(s_raw_datadir)
        ls_exp_entries = [
            os.path.join(s_raw_datadir, item) for item in ls_exp_entries
        ]
    def process_csv_paths_into_experiments(
        self,
        ls_csv_paths: List[str] = []
    ) -> List[Experiment]:
        if not isinstance(ls_csv_paths, list):
            raise ValueError(f"CSV paths must be a list: {ls_csv_paths}")

        if len(ls_csv_paths) == 0:
            raise ValueError("No csv paths provided")

        lo_exps = []
        for s_csv_path in ls_csv_paths:
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

    def process_exp_entries_with_subdirs(
        self,
        ls_exp_entries: List[str] = []
    ) -> List[str]:
        """
        Go through each experiment entry.
        If it has folders in it, then remove the entry and add the folders to the list.
        """
        if len(ls_exp_entries) == 0:
            raise ValueError("No experiment entries found")

        ls_exp_entries = [
            os.path.abspath(item) for item in ls_exp_entries
        ]

        ls_exp_entries_without_subdirs = []
        for s_exp_entry in ls_exp_entries:
            if s_exp_entry.endswith(".DS_Store"):
                continue

            if os.path.isfile(s_exp_entry):
                ls_exp_entries_without_subdirs.append(s_exp_entry)

            elif os.path.isdir(s_exp_entry):
                
                if not self.contains_dirs(s_exp_entry):

                    if self.contains_raw_files(s_exp_entry):
                        # NOTE: folders that contain pub_0.csv and sub_n.csv files
                        # - add the folders
                        ls_exp_entries_without_subdirs.append(s_exp_entry)

                    else:
                        # NOTE:folders that contain experiment csv files and 
                        # - add those experiment csv files
                        ls_exp_entries_without_subdirs.extend(
                            self.get_experiment_paths_from_fpath(s_exp_entry)
                        )

            else:
                raise ValueError(
                    f"Experiment entry is not a file or directory: {s_exp_entry}"
                )

        return ls_exp_entries_without_subdirs

    def contains_raw_files(self, s_exp_entry: str = "") -> bool:
        """
        Checks if the entry contains raw files (pub_0.csv or sub_n.csv).
        """
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")

        if not os.path.isdir(s_exp_entry):
            raise ValueError(f"Experiment entry is not a directory: {s_exp_entry}")

        ls_entries = os.listdir(s_exp_entry)
        for s_entry in ls_entries:
            if os.path.isfile(os.path.join(s_exp_entry, s_entry)):
                if self.is_raw_exp_file(os.path.join(s_exp_entry, s_entry)):
                    return True

        return False

    def contains_dirs(self, s_exp_entry: str = "") -> bool:
        """
        Checks if the entry contains directories.
        """
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")

        if not os.path.isdir(s_exp_entry):
            raise ValueError(f"Experiment entry is not a directory: {s_exp_entry}")

        ls_entries = os.listdir(s_exp_entry)
        for s_entry in ls_entries:
            if os.path.isdir(os.path.join(s_exp_entry, s_entry)):
                return True

        return False
                        
    def get_exp_with_expected_file_count(
        self,
        ld_exp_names_and_paths: List[Dict[str, List[str]]] = []
    ) -> List[Dict[str, List[str]]]:
        """
        Checks if there are the right number of pub and sub files according to 
        the experiment name.
        If it doesn't - it removes it from the list.
        """

        if len(ld_exp_names_and_paths) == 0:
            raise ValueError("No experiment names and paths provided")

        for d_exp_names_and_paths in ld_exp_names_and_paths:
            if 'name' not in d_exp_names_and_paths:
                raise ValueError("Experiment names and paths must have a name key")

            if 'paths' not in d_exp_names_and_paths:
                raise ValueError("Experiment names and paths must have a paths key")

            s_exp_name = d_exp_names_and_paths['name']
            ls_exp_paths = d_exp_names_and_paths['paths']

            if not self.follows_experiment_name_format(s_exp_name):
                s_exp_name = self.try_format_experiment_name(s_exp_name)
                if not self.follows_experiment_name_format(s_exp_name):
                    raise ValueError(
                        f"Experiment name does not follow expected format: {s_exp_name}"
                    )

            if len(ls_exp_paths) == 0:
                raise ValueError(f"No experiment paths found for {s_exp_name}")

            i_expected_file_count = self.get_expected_file_count(s_exp_name)
            i_actual_file_count = len(ls_exp_paths)

            if i_actual_file_count == 1:
                if s_exp_name not in ls_exp_paths[0]:
                    ld_exp_names_and_paths.remove(d_exp_names_and_paths)

            else:
                if i_actual_file_count != i_expected_file_count:
                    lg.warning(
                        "{} has {} files. Expected {}".format(
                            s_exp_name,
                            i_actual_file_count,
                            i_expected_file_count
                        )
                    )
                    ld_exp_names_and_paths.remove(d_exp_names_and_paths)

        return ld_exp_names_and_paths
                
    def get_expected_file_count(
        self,
        s_exp_name: str = ""
    ) -> int:
        """
        Get the expected file count for the experiment.
        Basically get the sub count and add 1.
        """
        if s_exp_name == "":
            raise Exception("No experiment name provided")

        if not self.follows_experiment_name_format(s_exp_name):
            s_exp_name = self.try_format_experiment_name(s_exp_name)
            if not self.follows_experiment_name_format(s_exp_name):
                raise ValueError(
                    f"Experiment name does not follow expected format: {s_exp_name}"
                )

        ls_parts = s_exp_name.split("_")
        if len(ls_parts) != 8:
            raise ValueError(f"Experiment name does not have 8 parts: {s_exp_name}")

        # Remove all non-numeric characters from ls_parts[3]
        i_sub_count = re.sub(r"[^0-9]", "", ls_parts[3])
        i_sub_count = int(i_sub_count)
        
        return i_sub_count + 1
                    
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
        
    def is_exp_name_in_dirpath(self, s_exp_path: str = "") -> bool:
        """
        Checks if the experiment name is in the directory path.
        """
        if s_exp_path == "":
            raise Exception("No experiment entry provided")

        i_slash_count = s_exp_path.count("/")
        if i_slash_count == 0:
            raise ValueError(
                f"Can't check for exp in dir of {s_exp_path}. There is no dir..."
            )

        s_dirname = os.path.basename(os.path.dirname(s_exp_path))
        return self.is_exp_name_in_str(s_dirname)
        
    def is_exp_name_in_filename(self, s_exp_path: str = "") -> bool:
        """
        Checks if the experiment name is in the filename.
        """
        if s_exp_path == "":
            raise Exception("No experiment entry provided")

        s_filename = os.path.basename(s_exp_path)

        return self.is_exp_name_in_str(s_filename)

    def is_exp_name_in_str(self, s_value: str = "") -> bool:
        """
        Checks if the experiment name is in the string.
        """
        if s_value == "":
            raise Exception("No experiment entry provided")

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
        
    def try_format_experiment_name_in_path(self, s_exp_entry: str = "") -> str:
        """
        Try to format the experiment name wherever it may be.
        """
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")

        if not self.is_path(s_exp_entry):
            raise ValueError(f"Experiment entry is not a path: {s_exp_entry}")

        b_exp_in_file = self.is_exp_name_in_filename(s_exp_entry)
        b_exp_in_dir = self.is_exp_name_in_dirpath(s_exp_entry)

        if not b_exp_in_file and not b_exp_in_dir:
            raise ValueError(
                f"Entry does not follow expected format in dir or file: {s_exp_entry}"
            )

        if b_exp_in_dir and b_exp_in_file:
            raise ValueError(
                f"So...both dir and file have the exp name: {s_exp_entry}."
            )
            
        if b_exp_in_file:
            s_filename = os.path.basename(s_exp_entry)
            
            s_dirname = os.path.basename(os.path.dirname(s_exp_entry))
            s_dir_dirpath = s_exp_entry.split(s_dirname)[0]

            s_filename = self.try_format_experiment_name(s_filename)

            return os.path.join(
                s_dir_dirpath,
                s_dirname,
                s_filename
            )

        else:
            s_filename = os.path.basename(s_exp_entry)
            s_dirname = os.path.basename(os.path.dirname(s_exp_entry))
            s_dir_dirpath = os.path.dirname(
                os.path.dirname(s_exp_entry)
            )
            s_dirname = self.try_format_experiment_name(s_dirname)

            return os.path.join(
                s_dir_dirpath,
                s_dirname,
                s_filename
            )

    def is_path(self, s_exp_entry: str = "") -> bool:
        """
        Checks if the entry is a path by counting the number of slashes.
        """
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")
        
        if not isinstance(s_exp_entry, str):
            raise ValueError(f"Experiment entry must be a string: {s_exp_entry}")

        i_slash_count = s_exp_entry.count("/")
        if i_slash_count == 0:
            return False

        return True
        
    def try_format_experiment_name(self, s_exp_name: str = "") -> str:
        if len(s_exp_name.split("_")) != 8:
            return s_exp_name

        s_exp_extension = ""
        if "." in s_exp_name:
            s_exp_extension = s_exp_name.split(".")[1]
            s_exp_name = s_exp_name.split(".")[0]
        
        s_exp_name = s_exp_name.upper()
        ls_sections = s_exp_name.split("_")

        if not ls_sections[0].endswith("SEC"):
            ls_sections[0] = ls_sections[0].replace("S", "SEC")

        if not ls_sections[2].endswith("PUB"):
            ls_sections[2] = ls_sections[2].replace("P", "PUB")

        if not ls_sections[3].endswith("SUB"):
            ls_sections[3] = ls_sections[3].replace("S", "SUB")

        s_exp_name = "_".join(ls_sections)

        if s_exp_extension != "":
            s_exp_name = f"{s_exp_name}.{s_exp_extension}"

        return s_exp_name
                        
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
        
    def get_experiment_paths_from_fpath(self, s_exp_entry: str = "") -> List[str]:
        """
        Get all files in the experiment directory.
        If the entry is a file, return that file.
        """
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")

        if not os.path.exists(s_exp_entry):
            raise Exception(f"Experiment entry does not exist: {s_exp_entry}")

        if os.path.isfile(s_exp_entry) and s_exp_entry.endswith(".csv"):
            return [s_exp_entry]

        ls_fpaths = self.recursively_get_fpaths(s_exp_entry)
        ls_csvpaths = [fpath for fpath in ls_fpaths if fpath.endswith(".csv")]

        return ls_csvpaths

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

    def old_create_dataset(self):
        lg.debug("Creating dataset...")

        exp_dirs = os.listdir(self.raw_datadir)
        if '.DS_Store' in exp_dirs:
            exp_dirs.remove('.DS_Store')

        exp_dirs = [
            exp_dir for exp_dir in exp_dirs \
                    if os.path.isdir(os.path.join(self.raw_datadir, exp_dir)) or
                    exp_dir.endswith(".csv")
        ]

        dataset_df = pd.DataFrame()
        exp_dirs = [os.path.join(self.raw_datadir, exp_dir) for exp_dir in exp_dirs]
        lg.debug(f"Found {len(exp_dirs)} experiment directories in {self.raw_datadir}")

        for index, exp_dir in enumerate(exp_dirs):
            lg.info(
                "[{}/{}] Processing {}".format(
                    index + 1,
                    len(exp_dirs),
                    os.path.basename(exp_dir)
                )
            )

            exp_name = os.path.basename(exp_dir)

            if exp_name.endswith(".csv"):
                exp_df = get_df_from_csv(exp_dir)
                exp_df['experiment_name'] = exp_name

            else:
                exp_df = pd.DataFrame()

                ls_fpaths = self.get_files_from_pathlib(Path(exp_dir))
                ls_csv_fpaths = [fpath for fpath in ls_fpaths if fpath.endswith(".csv")]

                if len(ls_csv_fpaths) == 0:
                    lg.warning(f"No csv files found in {exp_dir}")
                    continue

                csv_filenames = [os.path.basename(file) for file in ls_csv_fpaths]

                if "pub_0.csv" not in csv_filenames:
                    lg.warning(f"pub_0.csv not found in {exp_dir}")
                    continue

                exp = Experiment(exp_name)
                exp.set_pub_file([file for file in ls_csv_fpaths if "pub_0.csv" in file][0])
                exp.set_sub_files([file for file in ls_csv_fpaths if "sub_" in file])

                if not exp.validate_files():
                    continue

                exp.parse_pub_file()
                exp.parse_sub_files()

                if exp.get_subs_df() is None:
                    continue

                if exp.get_pub_df() is None:
                    continue

                qos = exp.get_qos()
                pub_df = exp.get_pub_df()
                sub_df = exp.get_subs_df()

                exp_df = pd.concat([sub_df, pub_df], axis=1)

                for key in qos.keys():
                    exp_df[key] = qos[key]

                # exp_df = calculate_averages(exp_df)
                exp_df['experiment_name'] = exp_name
                exp_df = aggregate_across_cols(exp_df, ['avg', 'total'])

            dataset_df = pd.concat([dataset_df, exp_df], ignore_index=True)

            if index % 10 == 0 and index != 0:
                if len(dataset_df) > 0:
                    dataset_df.to_parquet(dataset_path)
                    lg.info(f"Written to {dataset_path}")

        if len(dataset_df) == 0:
            lg.error(f"No data found in the dataset")

        dataset_df.to_parquet(dataset_path)
        lg.info(f"Dataset written to {dataset_path}")

        # Print how many experiments are in the dataset
        exp_count = dataset_df['experiment_name'].nunique()
        lg.info(f"Dataset has {exp_count} experiments")

    def generate_qos_exp_list(self):
        if not self.qos_variations:
            raise Exception("No qos_settings in config")

        if not isinstance(self.qos_variations, dict):
            raise ValueError(f"QoS config must be a dict: {self.qos_variations}")

        if self.qos_variations == {}:
            raise ValueError("QoS config must not be empty")

        required_keys = [
            'duration_secs',
            'datalen_bytes',
            'pub_count',
            'sub_count',
            'use_reliable',
            'use_multicast',
            'durability',
            'latency_count'
        ]

        keys = self.qos_variations.keys()
        if len(keys) == 0:
            raise ValueError("No options found for qos")

        for key in required_keys:
            if key not in self.qos_variations:
                raise ValueError(f"QoS config must have {key}")

        values = self.qos_variations.values()
        if len(values) == 0:
            raise ValueError("No values found for QoS")

        for value in values:
            if len(value) == 0:
                raise ValueError("One of the settings has no values.")

        combinations = list(itertools.product(*values))
        combination_dicts = [dict(zip(keys, combination)) for combination in combinations]

        if len(combination_dicts) == 0:
            raise ValueError(f"No combinations were generated from the QoS values:\n\t {self.qos_variations}")

        self.qos_exp_list = [get_qos_name(combination) for combination in combination_dicts]

    def get_missing_exps(self):
        self.generate_qos_exp_list()

        self.data_exp_list = [os.path.basename(path) for path in os.listdir(self.raw_datadir) if '.DS_Store' not in path]

        missing_tests = list(set(self.qos_exp_list) - set(self.data_exp_list))

        if len(missing_tests) == len(self.qos_exp_list):
            lg.warning("All tests are missing. The names are probably wrong.")
            self.data_exp_list = [item.replace("PUB", "P") for item in self.data_exp_list]
            self.data_exp_list = [item.replace("SUB", "S") for item in self.data_exp_list]
            missing_tests = list(set(self.qos_exp_list) - set(self.data_exp_list))

        lg.info("Expected {} experiments according to {}.".format(
            len(self.qos_exp_list), 
            self.apconf_path
        ))

        lg.info("Found {} experiments in {}.".format(
            len(self.data_exp_list),
            self.raw_datadir
        ))

        lg.info(f"Missing {len(missing_tests)} experiments.")

        return missing_tests

    def generate_missing_test_config(self, incomplete_exp_names=[]):
        if not self.config:
            raise Exception("No config")

        if 'gen_type' not in self.config.keys():
            raise Exception("No gen_type in config")
        
        if self.config['gen_type'] != "pcg":
            raise Exception("gen_type in the config is not pcg")

        if 'qos_settings' not in self.config.keys():
            raise Exception("No qos_settings in config")

        self.qos_variations = self.config['qos_settings']
        self.missing_exps = self.get_missing_exps()

        # Add missing_exps to incomplete_exp_names and remove duplicates
        self.missing_exps = list(set(incomplete_exp_names + self.missing_exps))
        fixed_exp_names = [exp.replace("PUB_", "P_") for exp in self.missing_exps]
        fixed_exp_names = [exp.replace("SUB_", "S_") for exp in fixed_exp_names]

        if len(self.missing_exps) > 0:
            lg.info(f"Generating missing test config with {len(self.missing_exps)} missing tests.")

        new_ap_config = self.config.copy()
        new_ap_config['experiment_names'] = fixed_exp_names
        new_ap_config = {'campaigns': [new_ap_config]}
        new_ap_config_path = self.apconf_path.replace(".toml", "_missing.toml")

        with open(new_ap_config_path, "w") as f:
            toml.dump(new_ap_config, f)

        lg.info(f"New apconf file generated at {new_ap_config_path}")

    def generate_config(self):
        lg.info("Generating config...")

        if not self.apconf_path:
            raise Exception("No apconf path provided")

        try:
            with open(self.apconf_path, "r") as f:
                ap_config = toml.load(f)

            if 'campaigns' not in ap_config:
                raise Exception("No campaigns section in apconf. You need [[campaigns]] section in apconf at the top.")

            campaign_configs = ap_config['campaigns']
            if len(campaign_configs) == 0:
                raise Exception("No campaigns defined in apconf")

            elif len(campaign_configs) > 1:
                lg.warning("Found multiple campaigns in {}. Using the first one.")

            self.config = ap_config['campaigns'][0]

        except Exception as e:
            raise e

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
