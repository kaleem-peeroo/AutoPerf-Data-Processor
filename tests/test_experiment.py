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
            assert os.path.basename(
                o_exp_run.s_run_name
            ) in ["run1_with_trailing_0", "run2_with_good_data"]
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

    def test_pick_best_run_with_raw(self):
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
        o_exp.pick_best_run()

        assert o_exp.best_exp_run is not None
        assert len(o_exp.lo_exp_runs) == 2
        assert o_exp.best_exp_run.s_exp_name == s_exp_name
        assert os.path.basename(
            o_exp.best_exp_run.s_run_name
        ) == "run1_with_trailing_0"

    def test_pick_best_run_with_exp_csv(self):
        s_test_dir = "./tests/data/test_experiment_with_runs_with_single_csv"
        s_exp_name = "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"
        o_exp = Experiment(
            s_name=s_exp_name,
            ls_csv_paths=[
                f"{s_test_dir}/run1/{s_exp_name}.csv",
                f"{s_test_dir}/run2/{s_exp_name}.csv",
            ],
        )

        o_exp.process_runs()
        o_exp.pick_best_run()

        assert o_exp.best_exp_run is not None
        assert len(o_exp.lo_exp_runs) == 2
        assert o_exp.best_exp_run.s_exp_name == s_exp_name
        assert os.path.basename(
            o_exp.best_exp_run.s_run_name
        ) == "run1"

    def test_get_good_exp_runs(self):
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
        lo_good_runs = o_exp.get_good_exp_runs()
        assert len(lo_good_runs) == 2

    def test_get_raw_exp_runs(self):
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
        lo_raw_runs = o_exp.get_raw_exp_runs()
        assert len(lo_raw_runs) == 2

    def test_sort_by_total_sample_count(self):
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
        lo_sorted_runs = o_exp.sort_by_total_sample_count(o_exp.lo_exp_runs)

        assert len(lo_sorted_runs) == 2
        assert lo_sorted_runs[0].get_total_sample_count() >= lo_sorted_runs[1].get_total_sample_count()

    def test_process(self):
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
        o_exp.pick_best_run()

        s_dpath = "./tests/output/test_process_summaries/"
        os.makedirs(s_dpath, exist_ok=True)

        df_summary = o_exp.process(s_dpath)

        assert os.path.exists(s_dpath)
        assert os.path.exists(
            os.path.join(s_dpath, f"{s_exp_name}.parquet")
        )

        assert isinstance(df_summary, pd.DataFrame)
        assert len(df_summary) > 0

        assert "avg_mbps_per_sub" in df_summary.columns
        assert "total_mbps_over_subs" in df_summary.columns
        assert "latency_us" in df_summary.columns
        assert "experiment_name" in df_summary.columns

        ls_input_cols = [
            'duration_secs',
            'datalen_bytes',
            'pub_count',
            'sub_count',
            'use_reliable',
            'use_multicast',
            'durability',
            'latency_count',
        ]
        for s_col in ls_input_cols:
            assert s_col in df_summary.columns

        # INFO: Delete the output directory after testing
        shutil.rmtree(s_dpath)

    def test_get_lat_df(self):
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

        # INFO: Normal Case - valid pub file
        s_pub_file = f"{S_PROJECT_PATH}/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run1_with_trailing_0/pub_0.csv"
        s_pub_file = ExperimentFile(
            s_exp_name,    
            s_pub_file,
        )
        sr = o_exp.get_lat_df(s_pub_file)
        assert isinstance(sr, pd.Series)
        assert len(sr) > 0
        assert sr.name == "latency_us"

    def test_get_mbps_df(self):
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

        # INFO: Normal Case - valid pub file
        s_pub_file = f"{S_PROJECT_PATH}/tests/data/test_experiment_with_runs_with_raw/600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC/run1_with_trailing_0/sub_0.csv"
        s_pub_file = ExperimentFile(
            s_exp_name,    
            s_pub_file,
        )
        sr = o_exp.get_mbps_df(s_pub_file)
        assert isinstance(sr, pd.Series)
        assert len(sr) > 0
        assert sr.name == "sub_0_mbps"

    def test_calculate_sub_metrics(self):
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
        o_exp.pick_best_run()

        df_summary = pd.DataFrame()
        for o_file in o_exp.best_exp_run.lo_exp_files:
            if not o_file.is_raw():
                df = o_file.get_df()
                df_summary = pd.concat([df_summary, df], axis=0)

            elif o_file.is_pub():
                df_lat = o_exp.get_lat_df(o_file)
                df_summary = pd.concat([df_summary, df_lat], axis=1)

            elif o_file.is_sub():
                df_mbps = o_exp.get_mbps_df(o_file)
                df_summary = pd.concat([df_summary, df_mbps], axis=1)

            else:
                raise ValueError("Unknown file type")

        df = o_exp.calculate_sub_metrics(df_summary)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "avg_mbps_per_sub" in df.columns
        assert "total_mbps_over_subs" in df.columns

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
        assert o_exp.is_valid_experiment_name(
            "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"
        ) is True

        # INFO: Normal Case - valid experiment name lower case
        assert o_exp.is_valid_experiment_name(
            "600sec_100b_10pub_1sub_rel_mc_0dur_100lc"
        ) is True

        # INFO: Normal Case - invalid experiment name
        assert o_exp.is_valid_experiment_name(
            "600S_100B_10P_1S_REL_MC_0DUR_100LC"
        ) is False

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
        assert o_exp.get_input_cols(
            "600SEC_100B_10PUB_1SUB_REL_MC_0DUR_100LC"
        ) == [
            {"duration_secs": 600},
            {"datalen_bytes": 100},
            {"pub_count": 10},
            {"sub_count": 1},
            {"use_reliable": True},
            {"use_multicast": True},
            {"durability": 0},
            {"latency_count": 100},
        ]
