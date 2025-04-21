import pytest
import os
import shutil
import pandas as pd

from rich.pretty import pprint
from pathlib import Path

from experiment_run import ExperimentRun

class TestExperimentRun:
    s_camp_dir_with_raw = "./tests/data/test_experiment_with_runs_with_raw"
    s_test_dir_with_raw = f"{s_camp_dir_with_raw}/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"

    d_run_with_good_data = {
        "exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "run_name": "run2_with_good_data",
        "csv_paths": [
            f"{s_test_dir_with_raw}/run2_with_good_data/pub_0.csv",
            f"{s_test_dir_with_raw}/run2_with_good_data/sub_0.csv",
        ],
    }
    
    d_run_with_trailing_0_good = {
        "exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "run_name": "run1_with_trailing_0",
        "csv_paths": [
            f"{s_test_dir_with_raw}/run1_with_trailing_0/pub_0.csv",
            f"{s_test_dir_with_raw}/run1_with_trailing_0/sub_0.csv",
        ],
    }

    d_run_with_trailing_0_bad = {
        "exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "run_name": "run3_with_trailing_0_bad_data",
        "csv_paths": [
            f"{s_test_dir_with_raw}/run3_with_trailing_0_bad_data/pub_0.csv",
            f"{s_test_dir_with_raw}/run3_with_trailing_0_bad_data/sub_0.csv",
        ],
    }

    def test_has_good_data_with_raw(self):
        # INFO: Normal Case - good data
        o_e = ExperimentRun(
            s_exp_name=self.d_run_with_good_data["exp_name"],
            s_run_name=self.d_run_with_good_data["run_name"],
            ls_csv_paths=self.d_run_with_good_data["csv_paths"],
        )
        assert o_e.has_good_data() is True

        # INFO: Normal Case - good data trailing 0
        o_e = ExperimentRun(
            s_exp_name=self.d_run_with_trailing_0_good["exp_name"],
            s_run_name=self.d_run_with_trailing_0_good["run_name"],
            ls_csv_paths=self.d_run_with_trailing_0_good["csv_paths"],
        )
        assert o_e.has_good_data() is True

        # INFO: Normal Case - bad data trailing 0
        o_e = ExperimentRun(
            s_exp_name=self.d_run_with_trailing_0_bad["exp_name"],
            s_run_name=self.d_run_with_trailing_0_bad["run_name"],
            ls_csv_paths=self.d_run_with_trailing_0_bad["csv_paths"],
        )
        assert o_e.has_good_data() is False

    def test_has_raw_data_with_raw(self):
        # INFO: Normal Case - good data
        o_e = ExperimentRun(
            s_exp_name=self.d_run_with_good_data["exp_name"],
            s_run_name=self.d_run_with_good_data["run_name"],
            ls_csv_paths=self.d_run_with_good_data["csv_paths"],
        )
        assert o_e.has_raw_data() is True

        # INFO: Normal Case - good data trailing 0
        o_e = ExperimentRun(
            s_exp_name=self.d_run_with_trailing_0_good["exp_name"],
            s_run_name=self.d_run_with_trailing_0_good["run_name"],
            ls_csv_paths=self.d_run_with_trailing_0_good["csv_paths"],
        )
        assert o_e.has_raw_data() is True

        # INFO: Normal Case - bad data trailing 0
        o_e = ExperimentRun(
            s_exp_name=self.d_run_with_trailing_0_bad["exp_name"],
            s_run_name=self.d_run_with_trailing_0_bad["run_name"],
            ls_csv_paths=self.d_run_with_trailing_0_bad["csv_paths"],
        )
        assert o_e.has_raw_data() is True
