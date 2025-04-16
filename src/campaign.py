import os
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

        self.ds_df = None

        self.config = None
        self.qos_variations = {}

        self.qos_exp_list   = []    # Expected experiment list generated from qos
        self.data_exp_list  = []    # experiment list from the data dir experiment names
        self.missing_exps   = []    # Missing experiment list

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

    def create_dataset(self):
        """
        1. Go through each item in the experiment directory
        2. Read the csv file
        3. Add the experiment name
        4. Add the qos settings to the dataframe
        5. Add to a big ds df
        6. Write the ds df to a parquet file

        The dataset should have the following columns:
            - experiment_name
            - latency_us
            - avg_mbps
            - total_mbps

        For any experiment, there should be:
            - around 600 avg_mbps samples
            - around 600 total_mbps samples
        """
        s_raw_datadir = self.get_raw_datadir()
        ld_exp_names_and_paths = self.get_experiments(s_raw_datadir)

        df_ds = pd.DataFrame()
        for d_exp_names_and_paths in ld_exp_names_and_paths:
            df_exp = self.process_exp_df(d_exp_names_and_paths)

    def process_exp_df(
        self, 
        d_exp_names_and_paths: Dict[str, List[str]] = {}
    ) -> pd.DataFrame:
        if not d_exp_names_and_paths:
            raise Exception("No experiment names and paths provided")

        if not isinstance(d_exp_names_and_paths, dict):
            raise ValueError(f"Experiment names and paths must be a dict: {d_exp_names_and_paths}")

        if 'name' not in d_exp_names_and_paths:
            raise ValueError("Experiment names and paths must have a name key")

        if 'paths' not in d_exp_names_and_paths:
            raise ValueError("Experiment names and paths must have a paths key")

        s_exp_name = d_exp_names_and_paths['name']
        ls_exp_paths = d_exp_names_and_paths['paths']

        df_exp = pd.DataFrame()
        for s_exp_path in ls_exp_paths:
            df_temp = self.get_exp_file_df(s_exp_path)
            df_exp = pd.concat([df_exp, df_temp], ignore_index=True)

        df_exp['experiment_name'] = s_exp_name
        df_exp = self.add_input_cols(df_exp)
            
        return df_exp

    def get_exp_file_df(
        self,
        s_exp_path: str = ""
    ) -> pd.DataFrame:
        """
        Get the experiment file dataframe.
        If its a raw file, process it.
        Otherwise, just read the csv file.
        """
        if self.is_raw_exp_file(s_exp_path):
            df_temp = self.process_file_df(s_exp_path)

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
        raise NotImplementedError("is_raw_exp_file not implemented")

    def process_file_df(
        self,
        s_exp_path: str = ""
    ) -> pd.DataFrame:
        """
        Checks if the file is a raw file or processed.
        """
        raise NotImplementedError("process_file_df not implemented")

    def add_input_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            raise Exception("No dataframe provided")

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a dataframe: {df}")

        if df.empty:
            raise ValueError("Input dataframe is empty")

        if 'experiment_name' not in df.columns:
            raise ValueError("Input dataframe must have experiment_name column")

        if df['experiment_name'].nunique() != 1:
            raise ValueError("Input dataframe must have only one experiment name")

        s_exp_name = df['experiment_name'].iloc[0]
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

        if not isinstance(s_exp_name, str):
            raise ValueError(f"Experiment name must be a string: {s_exp_name}")

        if "_" not in s_exp_name:
            raise ValueError(f"Experiment name must have underscores: {s_exp_name}")

        if not self.follows_experiment_name_format(s_exp_name):
            raise ValueError(f"Experiment name does not follow expected format: {s_exp_name}")

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
            d_qos[key] = int(d_qos[key])

        return d_qos

    def get_experiments(self, s_raw_datadir: str = ""):
        """
        Gather a list of experiments.
        In the format of {name: str, paths: list}.
        Name is experiment name.
        Paths is a list of all files for that experiment.
            This could be a single file or multiple files.
        """
        if s_raw_datadir == "":
            raise Exception("No raw data directory provided")

        if not os.path.exists(s_raw_datadir):
            raise Exception(f"Raw data directory does not exist: {s_raw_datadir}")

        if not os.path.isdir(s_raw_datadir):
            raise Exception(f"Raw data directory is not a directory: {s_raw_datadir}")

        ld_exp_names_and_paths = []
        ls_exp_entries = os.listdir(s_raw_datadir)
        ls_exp_entries = [
            os.path.join(s_raw_datadir, item) for item in ls_exp_entries
        ]

        for s_exp_entry in ls_exp_entries:
            s_exp_name = self.get_experiment_name_from_fpath(s_exp_entry)
            ls_exp_paths = self.get_experiment_paths_from_fpath(s_exp_entry)
            ld_exp_names_and_paths.append(
                {"name": s_exp_name, "paths": ls_exp_paths}
            )

        return ld_exp_names_and_paths

    def get_experiment_name_from_fpath(self, s_exp_entry: str = ""):
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")

        s_filename = os.path.basename(s_exp_entry).split(".")[0]
        s_dirname = os.path.basename(os.path.dirname(s_exp_entry))

        if self.follows_experiment_name_format(s_filename):
            return s_filename

        elif self.follows_experiment_name_format(s_dirname):
            return s_dirname

        else:
            raise ValueError(
                f"Experiment name does not follow expected format: {s_filename}"
            )
            
    def follows_experiment_name_format(self, s_filename: str = "") -> bool:
        """
        Checks if filename follows following format:
        *{int}SEC_{int}B_{int}P_{int}S_{REL/BE}_{MC/UC}_{int}DUR_{int}LC*
        """

        if s_filename == "":
            raise Exception("No filename provided")

        if not isinstance(s_filename, str):
            raise ValueError(f"Filename must be a string: {s_filename}")

        if s_filename == "":
            raise ValueError("Filename must not be empty")

        # INFO: Regex is hard to debug. Using manual checks.
        i_underscore_count = s_filename.count("_")
        if i_underscore_count != 7:
            lg.warning(f"Filename does not have 7 underscores: {s_filename}")
            return False

        if "." in s_filename:
            s_filename = s_filename.split(".")[0]

        ls_parts = s_filename.split("_")
        ld_end_matches = [
            {"values": ["SEC", "S"]},
            {"values": ["B"]},
            {"values": ["P", "PUB"]},
            {"values": ["S", "SUB"]},
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
                lg.warning(
                    f"Filename does not match expected format: {s_filename} ({part})"
                )
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

        ls_fpaths = self.recursively_get_fpaths(s_exp_entry)
        ls_csvpaths = [fpath for fpath in ls_fpaths if fpath.endswith(".csv")]

        return ls_csvpaths

    def recursively_get_fpaths(self, s_exp_entry: str = "") -> List[str]:
        if s_exp_entry == "":
            raise Exception("No experiment entry provided")

        if not os.path.exists(s_exp_entry):
            raise Exception(f"Experiment entry does not exist: {s_exp_entry}")

        if os.path.isfile(s_exp_entry):
            lg.warning("Experiment entry is a file. Returning that file.")
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
