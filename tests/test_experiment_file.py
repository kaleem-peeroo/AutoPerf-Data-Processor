import pytest
import os
import shutil
import pandas as pd

from rich.pretty import pprint
from pathlib import Path

from experiment_file import ExperimentFile

class TestExperimentFile:
    s_exp_fpath_valid = { 
        "s_exp_name": "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
        "s_path": "/Users/kaleem/PhD/Tools/AutoPerfDataProcessor/tests/data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv" 
    }

    s_pub_fpath_valid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": "/Users/kaleem/PhD/Tools/AutoPerfDataProcessor/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/pub_0.csv"
    }
    s_pub_fpath_invalid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": "/Users/kaleem/PhD/Tools/AutoPerfDataProcessor/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run4_with_trailing_0_bad_data_and_empty_pub/pub_0.csv"
    }

    s_sub_fpath_valid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": "/Users/kaleem/PhD/Tools/AutoPerfDataProcessor/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/sub_0.csv"
    }
    s_sub_fpath_invalid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": "/Users/kaleem/PhD/Tools/AutoPerfDataProcessor/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run3_with_trailing_0_bad_data/sub_0.csv"
    }

    def test_init_with_exp_file(self):
        o_e = ExperimentFile(
            s_exp_name="600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
            s_path="./tests/data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        )

        assert o_e.s_path == "./tests/data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"

    def test_init_with_pub_file(self):
        o_e = ExperimentFile(
            s_exp_name="600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
            s_path="./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/pub_0.csv"
        )

        assert o_e.s_path == "./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/pub_0.csv"

    def test_init_with_sub_file(self):
        o_e = ExperimentFile(
            s_exp_name="600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
            s_path="./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/sub_0.csv"
        )

        assert o_e.s_path == "./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/sub_0.csv"

    def test_is_raw(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        assert o_e.is_raw() is False

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        assert o_e.is_raw() is True

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        assert o_e.is_raw() is True

    def test_is_pub(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        assert o_e.is_pub() is False

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        assert o_e.is_pub() is True

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        assert o_e.is_pub() is False

    def test_is_sub(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        assert o_e.is_sub() is False

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        assert o_e.is_sub() is False

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        assert o_e.is_sub() is True
