import pytest
import os
import shutil
import pandas as pd

from rich.pretty import pprint
from pathlib import Path

from experiment_file import ExperimentFile

S_PROJECT_PATH = str(Path().cwd())


class TestExperimentFile:
    s_exp_fpath_valid = {
        "s_exp_name": "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
        "s_path": f"{S_PROJECT_PATH}/tests/data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv",
    }

    s_pub_fpath_valid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": f"{S_PROJECT_PATH}/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/pub_0.csv",
    }
    s_pub_fpath_invalid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": f"{S_PROJECT_PATH}/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run4_with_trailing_0_bad_data_and_empty_pub/pub_0.csv",
    }

    s_sub_fpath_valid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": f"{S_PROJECT_PATH}/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/sub_0.csv",
    }
    s_sub_fpath_invalid = {
        "s_exp_name": "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
        "s_path": f"{S_PROJECT_PATH}/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run3_with_trailing_0_bad_data/sub_0.csv",
    }

    def test_init_with_exp_file(self):
        o_e = ExperimentFile(
            s_exp_name="600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
            s_path="./tests/data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv",
        )

        assert (
            o_e.s_path
            == "./tests/data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        )

    def test_init_with_pub_file(self):
        o_e = ExperimentFile(
            s_exp_name="600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
            s_path="./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/pub_0.csv",
        )

        assert (
            o_e.s_path
            == "./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/pub_0.csv"
        )

    def test_init_with_sub_file(self):
        o_e = ExperimentFile(
            s_exp_name="600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC",
            s_path="./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/sub_0.csv",
        )

        assert (
            o_e.s_path
            == "./tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run2_with_good_data/sub_0.csv"
        )

    def test_is_raw(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        assert o_e.is_raw() is False

        # INFO: Normal Case - Extra text
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        assert o_e.is_raw() is False

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        assert o_e.is_raw() is True

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        assert o_e.is_raw() is True

        # INFO: Normal Case - pub csv with _output
        o_e = ExperimentFile(
            "300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC",
            "./tests/data/test_campaign_with_n_runs_dirs/300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC/run1/pub_0_output.csv",
        )
        assert o_e.is_raw() is True

        # INFO: Normal Case - sub csv with _output
        o_e = ExperimentFile(
            "300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC",
            "./tests/data/test_campaign_with_n_runs_dirs/300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC/run1/sub_0_output.csv",
        )
        assert o_e.is_raw() is True

    def test_is_pub(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        assert o_e.is_pub() is False

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        assert o_e.is_pub() is True

        # INFO: Normal Case - Sub csv
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

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        assert o_e.is_sub() is False

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        assert o_e.is_sub() is True

    def test_is_valid(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        assert o_e.is_valid() is True

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        assert o_e.is_valid() is True

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        assert o_e.is_valid() is True

        # INFO: Normal Case - Pub csv with invalid data
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_invalid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_invalid["s_path"],
        )
        with pytest.raises(ValueError):
            o_e.is_valid()

        # INFO: Normal Case - Sub csv with invalid data
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_invalid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_invalid["s_path"],
        )
        assert o_e.is_valid() is False

    def test_get_df(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        df = o_e.get_df()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 784
        assert df.shape[1] == 52

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        df = o_e.get_df()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 93
        assert df.shape[1] == 1

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        df = o_e.get_df()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 598
        assert df.shape[1] == 4

    def test_get_start_index(self):
        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        i_start_index = o_e.get_start_index()
        assert i_start_index == 2

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        i_start_index = o_e.get_start_index()
        assert i_start_index == 2

        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        with pytest.raises(ValueError):
            o_e.get_start_index()

    def test_get_end_index(self):
        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        i_end_index = o_e.get_end_index()
        assert i_end_index == 95

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        i_end_index = o_e.get_end_index()
        assert i_end_index == 600

        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        with pytest.raises(ValueError):
            o_e.get_end_index()

    def test_get_expected_sample_count(self):
        # INFO: Normal Case - Exp csv
        o_e = ExperimentFile(
            TestExperimentFile.s_exp_fpath_valid["s_exp_name"],
            TestExperimentFile.s_exp_fpath_valid["s_path"],
        )
        i_expected_samples = o_e.get_expected_sample_count()
        assert i_expected_samples == 600

        # INFO: Normal Case - Pub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )
        i_expected_samples = o_e.get_expected_sample_count()
        assert i_expected_samples == 600

        # INFO: Normal Case - Sub csv
        o_e = ExperimentFile(
            TestExperimentFile.s_sub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_sub_fpath_valid["s_path"],
        )
        i_expected_samples = o_e.get_expected_sample_count()
        assert i_expected_samples == 600

    def test_remove_trailing_zeroes(self):
        o_e = ExperimentFile(
            TestExperimentFile.s_pub_fpath_valid["s_exp_name"],
            TestExperimentFile.s_pub_fpath_valid["s_path"],
        )

        df_before = pd.DataFrame(
            {
                "a": [1, 0, 0, 0],
                "b": [2, 0, 0, 0],
            }
        )
        df_after = o_e.remove_trailing_zeroes(df_before)

        assert df_after["a"].tolist() == [1]
        assert df_after["b"].tolist() == [2]

    def test_parse_raw_file(self):
        o_file = ExperimentFile(
            "300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC",
            "./tests/data/test_campaign_with_n_runs_dirs/300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC/run2/sub_0.csv",
        )

        df = o_file.parse_raw_file()
        assert df is not None, "DF is None"
        assert len(df.columns) > 0, "No cols found in DF"
        assert (
            len(df.columns) == 4
        ), f"Did NOT find 8 cols in DF. Found {len(df.columns)}: {df.columns}"

        ls_wanted_cols = [
            "sub_0_sample_rate",
            "sub_0_mbps",
            "sub_0_lost_samples",
            "sub_0_lost_samples_percent",
        ]

        for s_wanted_col in ls_wanted_cols:
            assert (
                s_wanted_col in df.columns
            ), f"{s_wanted_col} not found in DF: {list(sorted(df.columns))}"

    def test_clean_df_col_names(self):
        o_file = ExperimentFile(
            "300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC",
            "./tests/data/test_campaign_with_n_runs_dirs/300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC/run2/sub_0.csv",
        )
        df_before = pd.DataFrame(
            {
                "    Mbps": [1, 2, 3],
                " Avg Samples/s": [1, 1, 1],
                "Length (Bytes)": [2, 2, 2],
                "Ave (μs)": [2, 2, 2],
                "Samples/s": [2, 2, 2],
            }
        )
        df_after = o_file.clean_df_col_names(df_before)
        assert len(df_after.columns) == 2
        assert set(df_after.columns) == set(["sub_0_mbps", "sub_0_sample_rate"])

        o_file = ExperimentFile(
            "300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC",
            "./tests/data/test_campaign_with_n_runs_dirs/300SEC_32B_1PUB_3SUB_BE_MC_0DUR_100LC/run2/pub_0.csv",
        )
        df_before = pd.DataFrame(
            {
                "Latency (us)": [1, 2, 3],
            }
        )
        df_after = o_file.clean_df_col_names(df_before)
        assert len(df_after.columns) == 1
        assert set(df_after.columns) == set(["latency_us"])
