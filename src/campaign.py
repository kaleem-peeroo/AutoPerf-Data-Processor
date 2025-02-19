import os
import toml
import itertools
import sys
import pandas as pd

from rich.pretty import pprint
from datetime import datetime

from logger import logger
from .utils import get_qos_name, calculate_averages, get_df_from_csv, aggregate_across_cols
from .experiment import Experiment

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class Campaign:
    def __init__(self, data_dir, apconf_path):
        self.data_dir       = data_dir
        self.apconf_path    = apconf_path
        self.config         = None
        self.qos_settings   = {}
        self.qos_exp_list   = []            # Expected experiment list generated from qos
        self.data_exp_list  = []            # experiment list from the data dir experiment names
        self.missing_exps   = []            # Missing experiment list

    def __rich_repr__(self):
        yield "data_dir", self.data_dir
        yield "apconf_path", self.apconf_path
        yield "config", self.config
        yield "qos_settings", self.qos_settings
        yield "qos_exp_list", self.qos_exp_list
        yield "data_exp_list", self.data_exp_list
        yield "missing_exps", self.missing_exps

    def get_data_dir(self):
        return self.data_dir

    def get_apconf_path(self):
        return self.apconf_path

    def set_data_dir(self, data_dir):
        self.data_dir = data_dir

    def set_apconf_path(self, apconf_path):
        self.apconf_path = apconf_path

    def validate_args(self):
        logger.info("Validating campaign args...")

        if not self.data_dir:
            logger.critical("No data dir provided")
            return False

        if not self.apconf_path:
            logger.critical("No apconf path provided")
            return False

        if "~" in self.data_dir:
            self.data_dir = os.path.expanduser(self.data_dir)

        if "~" in self.apconf_path:
            self.apconf_path = os.path.expanduser(self.apconf_path)

        if not os.path.exists(self.data_dir):
            logger.critical(f"Data dir does not exist: {self.data_dir}")
            return False

        if not os.path.exists(self.apconf_path):
            logger.critical(f"AP config path does not exist: {self.apconf_path}")
            return False

        if not os.path.isfile(self.apconf_path):
            logger.critical(f"AP config path is not a file: {self.apconf_path}")
            return False

        if not self.apconf_path.endswith(".toml"):
            logger.critical(f"AP config path does not end with .toml: {self.apconf_path}")
            return False

        return True

    def create_dataset(self, dataset_path):
        if not dataset_path.endswith(".parquet"):
            logger.warning(f"{dataset_path} does NOT end with .parquet. Appending .parquet to the end of the path")
            dataset_path = f"{dataset_path}.parquet"

        if not self.data_dir:
            raise Exception("No data dir")

        if not os.path.exists(self.data_dir):
            raise Exception(f"Data dir does not exist: {self.data_dir}")

        if len(os.listdir(self.data_dir)) == 0:
            raise Exception(f"Data dir is empty: {self.data_dir}")

        exp_dirs = os.listdir(self.data_dir)
        if '.DS_Store' in exp_dirs:
            exp_dirs.remove('.DS_Store')

        incomplete_exp_names = []

        dataset_df = pd.DataFrame()

        if os.path.exists(dataset_path):
            now_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dataset_path = f"{dataset_path.replace('.parquet', '')}_{now_timestamp}.parquet"
            logger.warning(f"Dataset already exists at {dataset_path}. Writing to {dataset_path}")

        exp_dirs = [os.path.join(self.data_dir, exp_dir) for exp_dir in exp_dirs]

        for index, exp_dir in enumerate(exp_dirs):
            logger.info(
                "[{}/{}] Processing {}".format(
                    index + 1,
                    len(exp_dirs),
                    os.path.basename(exp_dir)
                )
            )

            exp_name = os.path.basename(exp_dir)

            if exp_name.endswith(".csv"):
                exp_df = get_df_from_csv(exp_dir)

            else:
                exp_df = pd.DataFrame()

                csv_files = [os.path.join(
                    exp_dir, 
                    file
                ) for file in os.listdir(exp_dir) if file.endswith(".csv")]

                if len(csv_files) == 0:
                    logger.warning(f"No csv files found in {exp_dir}")
                    incomplete_exp_names.append(os.path.basename(exp_dir))
                    continue

                csv_filenames = [os.path.basename(file) for file in csv_files]

                if "pub_0.csv" not in csv_filenames:
                    logger.warning(f"pub_0.csv not found in {exp_dir}")
                    incomplete_exp_names.append(os.path.basename(exp_dir))
                    continue

                exp = Experiment(exp_name)
                exp.set_pub_file([file for file in csv_files if "pub_0.csv" in file][0])
                exp.set_sub_files([file for file in csv_files if "sub_" in file])

                if not exp.validate_files():
                    incomplete_exp_names.append(exp_name)
                    continue

                exp.parse_pub_file()
                exp.parse_sub_files()

                if exp.get_subs_df() is None:
                    incomplete_exp_names.append(exp_name)
                    continue

                if exp.get_pub_df() is None:
                    incomplete_exp_names.append(exp_name)
                    continue

                qos = exp.get_qos()
                pub_df = exp.get_pub_df()
                sub_df = exp.get_subs_df()

                exp_df = pd.concat([sub_df, pub_df], axis=1)
                exp_df['experiment_name'] = exp_name

                for key in qos.keys():
                    exp_df[key] = qos[key]

                # exp_df = calculate_averages(exp_df)
                exp_df = aggregate_across_cols(exp_df, ['avg', 'total'])

            dataset_df = pd.concat([dataset_df, exp_df], ignore_index=True)

            if index % 10 == 0 and index != 0:
                dataset_df.to_parquet(dataset_path)
                logger.info(f"Written to {dataset_path}")

        dataset_df.to_parquet(dataset_path)
        logger.info(f"Dataset written to {dataset_path}")

        # Print how many experiments are in the dataset
        exp_count = dataset_df['experiment_name'].nunique()
        logger.info(f"Dataset has {exp_count} experiments")
            
        return incomplete_exp_names

    def generate_qos_exp_list(self):
        if not self.qos_settings:
            raise Exception("No qos_settings in config")

        if not isinstance(self.qos_settings, dict):
            raise ValueError(f"QoS config must be a dict: {self.qos_settings}")

        if self.qos_settings == {}:
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

        keys = self.qos_settings.keys()
        if len(keys) == 0:
            raise ValueError("No options found for qos")

        for key in required_keys:
            if key not in self.qos_settings:
                raise ValueError(f"QoS config must have {key}")

        values = self.qos_settings.values()
        if len(values) == 0:
            raise ValueError("No values found for QoS")

        for value in values:
            if len(value) == 0:
                raise ValueError("One of the settings has no values.")

        combinations = list(itertools.product(*values))
        combination_dicts = [dict(zip(keys, combination)) for combination in combinations]

        if len(combination_dicts) == 0:
            raise ValueError(f"No combinations were generated from the QoS values:\n\t {self.qos_settings}")

        self.qos_exp_list = [get_qos_name(combination) for combination in combination_dicts]

    def get_missing_exps(self):
        self.generate_qos_exp_list()

        self.data_exp_list = [os.path.basename(path) for path in os.listdir(self.data_dir) if '.DS_Store' not in path]

        missing_tests = list(set(self.qos_exp_list) - set(self.data_exp_list))

        if len(missing_tests) == len(self.qos_exp_list):
            logger.warning("All tests are missing. The names are probably wrong.")
            self.data_exp_list = [item.replace("PUB", "P") for item in self.data_exp_list]
            self.data_exp_list = [item.replace("SUB", "S") for item in self.data_exp_list]
            missing_tests = list(set(self.qos_exp_list) - set(self.data_exp_list))

        logger.info("Expected {} experiments according to {}.".format(
            len(self.qos_exp_list), 
            self.apconf_path
        ))

        logger.info("Found {} experiments in {}.".format(
            len(self.data_exp_list),
            self.data_dir
        ))

        logger.info(f"Missing {len(missing_tests)} experiments.")

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

        self.qos_settings = self.config['qos_settings']
        self.missing_exps = self.get_missing_exps()

        # Add missing_exps to incomplete_exp_names and remove duplicates
        self.missing_exps = list(set(incomplete_exp_names + self.missing_exps))
        fixed_exp_names = [exp.replace("PUB_", "P_") for exp in self.missing_exps]
        fixed_exp_names = [exp.replace("SUB_", "S_") for exp in fixed_exp_names]

        if len(self.missing_exps) > 0:
            logger.info(f"Generating missing test config with {len(self.missing_exps)} missing tests.")

        new_ap_config = self.config.copy()
        new_ap_config['experiment_names'] = fixed_exp_names
        new_ap_config = {'campaigns': [new_ap_config]}
        new_ap_config_path = self.apconf_path.replace(".toml", "_missing.toml")

        with open(new_ap_config_path, "w") as f:
            toml.dump(new_ap_config, f)

        logger.info(f"New apconf file generated at {new_ap_config_path}")

    def generate_config(self):
        logger.info("Generating config...")

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
                logger.warning("Found multiple campaigns in {}. Using the first one.")

            self.config = ap_config['campaigns'][0]

        except Exception as e:
            raise e
