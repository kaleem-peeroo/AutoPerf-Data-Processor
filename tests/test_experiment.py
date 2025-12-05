from campaign import Campaign
import pytest
import os
import shutil
import pandas as pd

from rich.pretty import pprint
from pathlib import Path

from experiment import Experiment
from experiment_file import ExperimentFile

# Get the current working directory
s_cwd = Path(__file__).parent.resolve()
S_PROJECT_PATH = str(s_cwd.parent)


class TestExperiment:
    def test_process_runs_with_trailing_zeros(self):
        s_test_dir = "./tests/data/test_experiment_with_runs_with_raw"
        s_exp_name = "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"
        o_exp = Experiment(
            s_name=s_exp_name,
            ls_csv_paths=[
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/sub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/sub_0.csv",
            ],
        )

        o_exp.process_runs()

        assert len(o_exp.lo_exp_runs) == 2

        for i, o_exp_run in enumerate(o_exp.lo_exp_runs):
            assert o_exp_run.s_exp_name == s_exp_name
            assert os.path.basename(o_exp_run.s_run_name) in [
                "run1_with_trailing_0",
                "run2_with_good_data",
            ]
            assert len(o_exp_run.lo_exp_files) == 2

    def test_get_run_names(self):
        o_exp = Experiment(
            s_name="600SEC_1B_1PUB_1SUB_BE_MC_100LC",
            ls_csv_paths=[
                "run1/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
                "run2/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
                "run3/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
                "run4/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
            ],
        )

        ls_run_names = o_exp.get_run_names()

        assert len(ls_run_names) == 4
        assert ls_run_names == ["run1", "run2", "run3", "run4"]

    def test_get_run_names_with_n_runs_dirs(self):
        o_exp = Experiment(
            s_name="600SEC_1B_1PUB_1SUB_BE_MC_100LC",
            ls_csv_paths=[
                "run1/pub0.csv",
                "run1/sub0.csv",
                "run1/sub1.csv",
                "run1/sub2.csv",
                "run2/pub0.csv",
                "run2/sub0.csv",
                "run2/sub1.csv",
                "run2/sub2.csv",
                "run3/pub0.csv",
                "run3/sub0.csv",
                "run3/sub1.csv",
                "run3/sub2.csv",
            ],
        )

        ls_run_names = o_exp.get_run_names()

        assert len(ls_run_names) == 3
        assert ls_run_names == ["run1", "run2", "run3"]

    def test_summarise(self):
        o_campaign = Campaign(
            {
                "exp_folders": "./tests/data/test_campaign_with_dirs_simple/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_dirs_simple.parquet",
            }
        )
        o_exp = o_campaign.gather_experiments(o_campaign.get_raw_datadir())[0]
        o_exp.summarise(o_campaign.s_summaries_dpath)

        assert os.path.exists(
            os.path.join(
                o_campaign.s_summaries_dpath,
                "300SEC_1B_1PUB_3SUB_BE_MC_0DUR_100LC.parquet",
            )
        )

    def test_format_exp_name(self):
        s_test_dir = "./tests/data/test_experiment_with_runs_with_raw"
        s_exp_name = "600S_100B_10P_1S_REL_MC_0DUR_100LC"
        o_exp = Experiment(
            s_name=s_exp_name,
            ls_csv_paths=[
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/sub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/sub_0.csv",
            ],
        )

        formatted_name = o_exp.format_exp_name(s_exp_name)
        assert formatted_name == "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"

    def test_is_valid_experiment_name(self):
        s_test_dir = "./tests/data/test_experiment_with_runs_with_raw"
        s_exp_name = "600S_100B_10P_1S_REL_MC_0DUR_100LC"
        o_exp = Experiment(
            s_name=s_exp_name,
            ls_csv_paths=[
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/sub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/sub_0.csv",
            ],
        )

        # INFO: Normal Case - valid experiment name
        assert (
            o_exp.is_valid_experiment_name("600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC")
            is True
        )

        # INFO: Normal Case - valid experiment name lower case
        assert (
            o_exp.is_valid_experiment_name("600sec_100b_10pub_1sub_rel_mc_0dur_100lc")
            is True
        )

        # INFO: Normal Case - invalid experiment name
        assert (
            o_exp.is_valid_experiment_name("600S_100B_10P_1S_REL_MC_0DUR_100LC")
            is False
        )

    def test_add_input_cols(self):
        s_test_dir = "./tests/data/test_experiment_with_runs_with_raw"
        s_exp_name = "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"
        o_exp = Experiment(
            s_name=s_exp_name,
            ls_csv_paths=[
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/sub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/sub_0.csv",
            ],
        )

        df_test = pd.DataFrame(
            {
                "latency_us": [100, 200, 300],
                "avg_mbps_per_sub": [10, 20, 30],
                "total_mbps_over_subs": [100, 200, 300],
                "experiment_name": ["600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"] * 3,
            }
        )

        df_test = o_exp.add_input_cols(df_test)
        assert isinstance(df_test, pd.DataFrame)
        assert len(df_test) == 3

        # INFO: Keep the original columns
        assert "latency_us" in df_test.columns
        assert "avg_mbps_per_sub" in df_test.columns
        assert "total_mbps_over_subs" in df_test.columns
        assert "experiment_name" in df_test.columns

        ls_input_cols = [
            "duration_secs",
            "datalen_bytes",
            "pub_count",
            "sub_count",
            "use_reliable",
            "use_multicast",
            "durability",
            "latency_count",
        ]
        for s_col in ls_input_cols:
            assert s_col in df_test.columns

    def test_get_input_cols(self):
        s_test_dir = "./tests/data/test_experiment_with_runs_with_raw"
        s_exp_name = "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"
        o_exp = Experiment(
            s_name=s_exp_name,
            ls_csv_paths=[
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run1_with_trailing_0/sub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/pub_0.csv",
                f"{s_test_dir}/{s_exp_name}/run2_with_good_data/sub_0.csv",
            ],
        )

        # INFO: Normal Case
        assert o_exp.get_input_cols("600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC") == [
            {"duration_secs": 600},
            {"datalen_bytes": 100},
            {"pub_count": 10},
            {"sub_count": 1},
            {"use_reliable": True},
            {"use_multicast": True},
            {"durability": 0},
            {"latency_count": 100},
        ]
