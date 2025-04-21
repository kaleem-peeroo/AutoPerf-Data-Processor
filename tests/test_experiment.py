import pytest
import os
import shutil
import pandas as pd

from rich.pretty import pprint
from pathlib import Path

from experiment import Experiment

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
