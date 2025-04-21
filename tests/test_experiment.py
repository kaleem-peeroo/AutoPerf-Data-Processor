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
        s_exp_name = "600SEC_1B_1PUB_1SUB_BE_MC_100LC"
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
            assert len(o_exp_run.ls_csv_paths) == 2

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

        pprint(ls_run_names)

        assert len(ls_run_names) == 4
        assert ls_run_names == ["run1", "run2", "run3", "run4"]
