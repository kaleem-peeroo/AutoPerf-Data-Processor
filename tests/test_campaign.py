import pytest
import os
import shutil
import pandas as pd

from rich.pretty import pprint
from pathlib import Path

from campaign import Campaign
from experiment import Experiment
from tests.configs.normal import LD_DATASETS


class TestCampaign:
    def test_init_with_normal_case(self):
        d_ds = {
            "name": "test campaign with dirs",
            "exp_folders": "./tests/data/test_campaign_with_dirs_small/",
            "ap_config": "",
            "dataset_path": "./tests/output/test_campaign_with_dirs_small.parquet",
        }

        o_c = Campaign(d_ds)
        assert o_c is not None
        assert isinstance(o_c, Campaign)

        assert o_c.ds_output_path == d_ds["dataset_path"]
        assert o_c.apconf_path == d_ds["ap_config"]
        assert o_c.raw_datadir == d_ds["exp_folders"]
        assert o_c.df_ds is None
        assert (
            o_c.s_summaries_dpath
            == "./tests/output/test_campaign_with_dirs_small_summaries"
        )

    def test_summarise_experiments(self):
        d_config = {
            "name": "test campaign with dirs",
            "exp_folders": "./tests/data/test_campaign_with_dirs_small/",
            "ap_config": "",
            "dataset_path": "./tests/output/test_campaign_with_dirs_small.parquet",
        }

        o_c = Campaign(d_config)
        o_c.summarise_experiments()
        assert os.path.exists(o_c.s_summaries_dpath)
        assert os.path.isdir(o_c.s_summaries_dpath)
        assert (
            len(os.listdir(o_c.s_summaries_dpath)) == 2
        ), f"Expected 2 summaries, got {len(os.listdir(o_c.s_summaries_dpath))}"

        # Delete the generated summaries folder at the end of the test
        shutil.rmtree(o_c.s_summaries_dpath)

    def test_summarise_experiments_with_n_runs(self):
        d_config = {
            "name": "test campaign with n runs and dirs",
            "exp_folders": "./tests/data/test_campaign_with_n_runs_dirs/",
            "ap_config": "",
            "dataset_path": "./tests/output/test_campaign_with_n_runs.parquet",
        }

        if os.path.exists(d_config["dataset_path"]):
            os.remove(d_config["dataset_path"])

        if os.path.exists(d_config["dataset_path"].replace(".parquet", "_summaries/")):
            shutil.rmtree(d_config["dataset_path"].replace(".parquet", "_summaries/"))

        o_c = Campaign(d_config)
        o_c.summarise_experiments()
        assert os.path.exists(o_c.s_summaries_dpath)
        assert os.path.isdir(o_c.s_summaries_dpath)
        assert (
            len(os.listdir(o_c.s_summaries_dpath)) == 2
        ), f"Expected 2 summaries, got {len(os.listdir(o_c.s_summaries_dpath))}"

        # INFO: Validate the summarised files
        for s_summary_path in os.listdir(o_c.s_summaries_dpath):
            s_summary_path = os.path.join(o_c.s_summaries_dpath, s_summary_path)
            df_summary = pd.read_parquet(s_summary_path)

            ls_existing_cols = sorted(list(df_summary.columns))
            ls_wanted_cols = [
                "experiment_name",
                "run_n",
                "latency_us",
                "avg_mbps_per_sub",
                "total_mbps_over_subs",
                "duration_secs",
                "datalen_bytes",
                "pub_count",
                "sub_count",
                "use_reliable",
                "use_multicast",
                "durability",
                "latency_count",
            ]
            for s_col in ls_wanted_cols:
                assert (
                    s_col in ls_existing_cols
                ), f"{s_col} NOT found in summarised dataset: {ls_existing_cols}"

        # INFO: Delete the generated summaries folder at the end of the test
        shutil.rmtree(o_c.s_summaries_dpath)

    def test_create_dataset_with_dirs(self):
        d_config = {
            "name": "test campaign with dirs",
            "exp_folders": "./tests/data/test_campaign_with_dirs_small/",
            "ap_config": "",
            "dataset_path": "./tests/output/test_campaign_with_dirs_small.parquet",
        }

        o_c = Campaign(d_config)
        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        o_c.summarise_experiments()
        o_c.create_dataset()

        assert os.path.exists(o_c.ds_output_path)
        assert os.path.isfile(o_c.ds_output_path)
        assert os.path.getsize(o_c.ds_output_path) > 0
        assert o_c.df_ds is not None
        assert isinstance(o_c.df_ds, pd.DataFrame)
        assert len(o_c.df_ds) > 0
        assert len(o_c.df_ds.columns) > 0

        ls_wanted_cols = [
            "experiment_name",
            "latency_us",
            "avg_mbps_per_sub",
            "total_mbps_over_subs",
        ]
        for s_col in ls_wanted_cols:
            assert s_col in o_c.df_ds.columns

        df = o_c.get_df_ds()
        ls_exp_names = df["experiment_name"].unique().tolist()

        for o_exp in lo_exps:
            s_exp_name = o_exp.format_exp_name(o_exp.get_name())
            assert (
                s_exp_name in ls_exp_names
            ), f"Experiment name {s_exp_name} not found in df: {ls_exp_names}"

            df_exp = df[df["experiment_name"] == s_exp_name].copy()
            assert len(df_exp) > 0
            assert len(df_exp.columns) > 0
            assert df_exp["experiment_name"].nunique() == 1
            assert df_exp["experiment_name"].iloc[0] == s_exp_name

            # INFO: No duplicate columns
            ls_cols = df_exp.columns.tolist()
            assert len(ls_cols) == len(set(ls_cols))

            # INFO: Check latency_us
            df_exp_lat = df_exp[["latency_us"]].copy()
            df_exp_lat.dropna(inplace=True)
            assert len(df_exp_lat) > 0

            # INFO: Check avg_mbps
            df_exp_avg_mbps = df_exp[["avg_mbps_per_sub"]].copy()
            df_exp_avg_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_avg_mbps) > 400
            assert len(df_exp_avg_mbps) < 800

            # INFO: Check total_mbps
            df_exp_total_mbps = df_exp[["total_mbps_over_subs"]].copy()
            df_exp_total_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_total_mbps) > 400
            assert len(df_exp_total_mbps) < 800

        # Delete the summaries
        shutil.rmtree(o_c.s_summaries_dpath)
        # Delete the dataset
        Path(o_c.ds_output_path).unlink()

    def test_create_dataset_with_csv(self):
        d_config = {
            "name": "test campaign with csv",
            "exp_folders": "./tests/data/test_campaign_with_csv/",
            "ap_config": "",
            "dataset_path": "./tests/output/test_campaign_with_csv_small.parquet",
        }

        o_c = Campaign(d_config)
        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        o_c.summarise_experiments()
        o_c.create_dataset()

        assert os.path.exists(o_c.ds_output_path)
        assert os.path.isfile(o_c.ds_output_path)
        assert os.path.getsize(o_c.ds_output_path) > 0
        assert o_c.df_ds is not None
        assert isinstance(o_c.df_ds, pd.DataFrame)
        assert len(o_c.df_ds) > 0
        assert len(o_c.df_ds.columns) > 0

        ls_wanted_cols = [
            "experiment_name",
            "latency_us",
            "avg_mbps_per_sub",
            "total_mbps_over_subs",
        ]
        for s_col in ls_wanted_cols:
            assert (
                s_col in o_c.df_ds.columns
            ), f"Column {s_col} not found in df: {o_c.df_ds.columns}"

        df = o_c.get_df_ds()
        ls_exp_names = df["experiment_name"].unique().tolist()

        for o_exp in lo_exps:
            s_exp_name = o_exp.format_exp_name(o_exp.get_name())
            assert s_exp_name in ls_exp_names

            df_exp = df[df["experiment_name"] == s_exp_name].copy()
            assert len(df_exp) > 0
            assert len(df_exp.columns) > 0
            assert df_exp["experiment_name"].nunique() == 1
            assert df_exp["experiment_name"].iloc[0] == s_exp_name

            # INFO: No duplicate columns
            ls_cols = df_exp.columns.tolist()
            assert len(ls_cols) == len(set(ls_cols))

            # INFO: Check latency_us
            df_exp_lat = df_exp[["latency_us"]].copy()
            df_exp_lat.dropna(inplace=True)
            assert len(df_exp_lat) > 0

            # INFO: Check avg_mbps
            df_exp_avg_mbps = df_exp[["avg_mbps_per_sub"]].copy()
            df_exp_avg_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_avg_mbps) > 400
            assert len(df_exp_avg_mbps) < 800

            # INFO: Check total_mbps
            df_exp_total_mbps = df_exp[["total_mbps_over_subs"]].copy()
            df_exp_total_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_total_mbps) > 400
            assert len(df_exp_total_mbps) < 800

        # Delete the summaries
        shutil.rmtree(o_c.s_summaries_dpath)
        # Delete the dataset
        Path(o_c.ds_output_path).unlink()

    def test_get_sub_mbps_cols(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with 1 sub mbps col
        df = pd.DataFrame(
            {
                "sub_0_mbps": [1, 2, 3],
                "latency_us": [7, 8, 9],
            }
        )
        assert o_c.get_sub_mbps_cols(df) == ["sub_0_mbps"]

        # INFO: Normal Case - with 2 sub mbps cols
        df = pd.DataFrame(
            {
                "sub_0_mbps": [1, 2, 3],
                "sub_1_mbps": [4, 5, 6],
                "latency_us": [7, 8, 9],
            }
        )
        assert o_c.get_sub_mbps_cols(df) == ["sub_0_mbps", "sub_1_mbps"]

        # INFO: Empty Case - no sub mbps cols
        df = pd.DataFrame(
            {
                "latency_us": [7, 8, 9],
            }
        )
        assert o_c.get_sub_mbps_cols(df) == []

    def test_gather_experiments_with_csv(self):
        o_c = Campaign(
            {
                "name": "another test campaign with csv",
                "exp_folders": "./tests/data/test_campaign_with_csv/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_csv.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        assert lo_exps is not None
        assert isinstance(lo_exps, list)

        # There are 4 csv files (1 per experiment)
        assert len(lo_exps) == 4

        for o_exp in lo_exps:
            assert isinstance(o_exp, Experiment)
            assert o_exp.s_name is not None
            assert isinstance(o_exp.s_name, str)
            assert o_exp.ls_csv_paths is not None
            assert isinstance(o_exp.ls_csv_paths, list)
            assert len(o_exp.ls_csv_paths) > 0
            assert len(o_exp.lo_exp_runs) == 1

    def test_gather_experiments_with_dir_with_csv(self):
        o_c = Campaign(
            {
                "name": "another test campaign with dir with csv",
                "exp_folders": "./tests/data/test_campaign_with_dir_with_csv/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_dir_with_csv.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        assert lo_exps is not None
        assert isinstance(lo_exps, list)

        # There are 18 csv files in 1 dir (1 per experiment)
        assert len(lo_exps) == 18

        for o_exp in lo_exps:
            assert isinstance(o_exp, Experiment)
            assert o_exp.s_name is not None
            assert isinstance(o_exp.s_name, str)
            assert o_exp.ls_csv_paths is not None
            assert isinstance(o_exp.ls_csv_paths, list)
            assert len(o_exp.ls_csv_paths) > 0

    def test_gather_experiments_with_dirs_simple(self):
        o_c = Campaign(
            {
                "name": "another test campaign with dirs",
                "exp_folders": "./tests/data/test_campaign_with_dirs_simple/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_dirs_simple.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        assert lo_exps is not None
        assert isinstance(lo_exps, list)

        # There 1 experiment with 4 files
        assert len(lo_exps) == 1

        for o_exp in lo_exps:
            assert isinstance(o_exp, Experiment)
            assert o_exp.s_name is not None
            assert isinstance(o_exp.s_name, str)
            assert o_exp.ls_csv_paths is not None
            assert isinstance(o_exp.ls_csv_paths, list)
            assert len(o_exp.ls_csv_paths) > 0

    def test_gather_experiments_with_dirs_small(self):
        o_c = Campaign(
            {
                "name": "another test campaign with dirs small",
                "exp_folders": "./tests/data/test_campaign_with_dirs_small/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_dirs_small.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        assert lo_exps is not None
        assert isinstance(lo_exps, list)

        # There 2 experiments
        assert len(lo_exps) == 2

        for o_exp in lo_exps:
            assert isinstance(o_exp, Experiment)
            assert o_exp.s_name is not None
            assert isinstance(o_exp.s_name, str)
            assert o_exp.ls_csv_paths is not None
            assert isinstance(o_exp.ls_csv_paths, list)
            assert len(o_exp.ls_csv_paths) > 0

    def test_gather_experiments_with_errors(self):
        o_c = Campaign(
            {
                "name": "another test campaign with errors",
                "exp_folders": "./tests/data/test_campaign_with_errors/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_errors.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        assert lo_exps is not None
        assert isinstance(lo_exps, list)

        # There 2 experiments
        assert len(lo_exps) == 2

        for o_exp in lo_exps:
            assert isinstance(o_exp, Experiment)
            assert o_exp.s_name is not None
            assert isinstance(o_exp.s_name, str)
            assert o_exp.ls_csv_paths is not None
            assert isinstance(o_exp.ls_csv_paths, list)
            assert len(o_exp.ls_csv_paths) > 0

    def test_gather_experiments_with_mix(self):
        o_c = Campaign(
            {
                "name": "another test campaign with mix",
                "exp_folders": "./tests/data/test_campaign_with_mix/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_mix.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        lo_exps = o_c.gather_experiments(s_raw_datadir)

        assert lo_exps is not None
        assert isinstance(lo_exps, list)

        # There 5 experiments
        assert len(lo_exps) == 5

        for o_exp in lo_exps:
            assert isinstance(o_exp, Experiment)
            assert o_exp.s_name is not None
            assert isinstance(o_exp.s_name, str)
            assert o_exp.ls_csv_paths is not None
            assert isinstance(o_exp.ls_csv_paths, list)
            assert len(o_exp.ls_csv_paths) > 0

    def test_process_csv_paths_into_experiments_with_csv(self):
        o_c = Campaign(
            {
                "name": "test campaign with csv",
                "exp_folders": "./tests/data/test_campaign_with_csv/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_csv.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        ls_fpaths = o_c.recursively_get_fpaths(s_raw_datadir)
        ls_csv_paths = [_ for _ in ls_fpaths if _.endswith(".csv")]
        # There are 4 csv files (1 per experiment)
        assert len(ls_csv_paths) == 4

        lo_exps = o_c.process_csv_paths_into_experiments(ls_csv_paths)

        assert lo_exps is not None
        assert isinstance(lo_exps, list)

        assert len(lo_exps) == 4
        for o_exp in lo_exps:
            assert isinstance(o_exp, Experiment)
            assert o_exp.s_name is not None
            assert isinstance(o_exp.s_name, str)
            assert o_exp.ls_csv_paths is not None
            assert isinstance(o_exp.ls_csv_paths, list)
            assert len(o_exp.ls_csv_paths) > 0

    def test_process_csv_paths_into_experiments_with_dir_with_csv(self):
        o_c = Campaign(
            {
                "name": "test campaign with dir with csv",
                "exp_folders": "./tests/data/test_campaign_with_dir_with_csv/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_dir_with_csv.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        ls_fpaths = o_c.recursively_get_fpaths(s_raw_datadir)
        ls_csv_paths = [_ for _ in ls_fpaths if _.endswith(".csv")]
        # There are 18 csv files in 1 dir (1 per experiment)
        assert len(ls_csv_paths) == 18

    def test_process_csv_paths_into_experiments_with_dirs_simple(self):
        o_c = Campaign(
            {
                "name": "test campaign with dirs simple",
                "exp_folders": "./tests/data/test_campaign_with_dirs_simple/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_dirs_simple.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        ls_fpaths = o_c.recursively_get_fpaths(s_raw_datadir)
        ls_csv_paths = [_ for _ in ls_fpaths if _.endswith(".csv")]
        # There are 4 csv files in 1 dir (1 per experiment)
        assert len(ls_csv_paths) == 4

    def test_process_csv_paths_into_experiments_with_dirs_small(self):
        o_c = Campaign(
            {
                "name": "test campaign with dirs small",
                "exp_folders": "./tests/data/test_campaign_with_dirs_small/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_dirs_small.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        ls_fpaths = o_c.recursively_get_fpaths(s_raw_datadir)
        ls_csv_paths = [_ for _ in ls_fpaths if _.endswith(".csv")]
        # There are 8 csv files in 2 dir (4 per experiment)
        assert len(ls_csv_paths) == 8

    def test_process_csv_paths_into_experiments_with_errors(self):
        o_c = Campaign(
            {
                "name": "test campaign with dirs errors",
                "exp_folders": "./tests/data/test_campaign_with_errors/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_errors.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        ls_fpaths = o_c.recursively_get_fpaths(s_raw_datadir)
        ls_csv_paths = [_ for _ in ls_fpaths if _.endswith(".csv")]
        # There are 4 csv files in 2 dir (2 per experiment)
        assert len(ls_csv_paths) == 4

    def test_process_csv_paths_into_experiments_with_mix(self):
        o_c = Campaign(
            {
                "name": "test campaign with dirs mix",
                "exp_folders": "./tests/data/test_campaign_with_mix/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign_with_mix.parquet",
            }
        )

        s_raw_datadir = o_c.get_raw_datadir()
        ls_fpaths = o_c.recursively_get_fpaths(s_raw_datadir)
        ls_csv_paths = [_ for _ in ls_fpaths if _.endswith(".csv")]

        # There are 77 csv files
        assert len(ls_csv_paths) == 77

    def test_get_experiment_name_from_fpath(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case: Name in file
        s_exp_name = o_c.get_experiment_name_from_fpath(
            "./data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        )

        assert isinstance(s_exp_name, str)
        assert len(s_exp_name) > 0
        assert s_exp_name == "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"

        # INFO: Normal Case: Name in directory
        s_exp_name = o_c.get_experiment_name_from_fpath(
            "./test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC/pub_0.csv"
        )
        assert isinstance(s_exp_name, str)
        assert len(s_exp_name) > 0
        assert s_exp_name == "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"

        # INFO: Error Case: No name
        with pytest.raises(ValueError):
            s_exp_name = o_c.get_experiment_name_from_fpath(
                "./test_campaign_with_csv/pub_0.csv"
            )

        # INFO: Normal Case: Name in dir
        s_exp_name = o_c.get_experiment_name_from_fpath(
            "./test_campaign_with_dir_with_csv/120SEC_100B_1PUB_1SUB_REL_MC_0DUR_100LC.csv"
        )
        assert isinstance(s_exp_name, str)
        assert len(s_exp_name) > 0
        assert s_exp_name == "120SEC_100B_1PUB_1SUB_REL_MC_0DUR_100LC"

    def test_follows_experiment_name_format(self):
        o_c = Campaign(LD_DATASETS[0])

        ls_normal_cases = [
            # INFO: Normal case - most up to date format
            "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
            # INFO: REL instead of BE
            "600SEC_100B_15PUB_15SUB_REL_MC_3DUR_100LC",
            # INFO: UC instead of MC
            "600SEC_100B_15PUB_15SUB_BE_UC_3DUR_100LC",
            # INFO: 0DUR instead of 3DUR
            "600SEC_100B_15PUB_15SUB_BE_MC_0DUR_100LC",
            # INFO: With .csv at end
            "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv",
        ]

        for s_exp_name in ls_normal_cases:
            b_follows_format = o_c.follows_experiment_name_format(s_exp_name)
            assert isinstance(b_follows_format, bool)
            assert b_follows_format is True, f"Failed for {s_exp_name}"

        # INFO: Error Case - file path to valid file
        s_exp_name = "./tests/data/test_campaign_with_dirs_small/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        assert o_c.follows_experiment_name_format(s_exp_name) is False

    def test_get_experiment_name_from_fpath(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with old format name in filename
        assert (
            o_c.get_experiment_name_from_fpath(
                "./data/campaign/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
            )
            == "300SEC_1B_1P_3S_BE_MC_0DUR_100LC"
        )

        # INFO: Normal Case - with new format name in filename
        assert (
            o_c.get_experiment_name_from_fpath(
                "./data/campaign/300SEC_1B_1PUB_3SUB_BE_MC_0DUR_100LC/pub_0.csv"
            )
            == "300SEC_1B_1PUB_3SUB_BE_MC_0DUR_100LC"
        )

        # INFO: Normal Case - name far from file
        assert (
            o_c.get_experiment_name_from_fpath(
                "./300SEC_1B_1P_3S_BE_MC_0DUR_100LC/a/b/c/d/pub_0.csv"
            )
            == "300SEC_1B_1P_3S_BE_MC_0DUR_100LC"
        )

        # INFO: Normal Case - name in file
        assert (
            o_c.get_experiment_name_from_fpath("./300SEC_1B_1P_3S_BE_MC_0DUR_100LC.csv")
            == "300SEC_1B_1P_3S_BE_MC_0DUR_100LC"
        )

        # INFO: Normal Case - no name
        with pytest.raises(ValueError):
            o_c.get_experiment_name_from_fpath("./data/campaign/some_random_name.csv")

    def test_is_experiment_name_in_fpath(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with name in filename
        assert (
            o_c.is_experiment_name_in_fpath(
                "./data/campaign/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
            )
            is True
        )

        # INFO: Normal Case - with latest name format in filename
        assert (
            o_c.is_experiment_name_in_fpath(
                "./data/campaign/300SEC_1B_1PUB_3SUB_BE_MC_0DUR_100LC/pub_0.csv"
            )
            is True
        )

        # INFO: Normal Case - with name in dir
        assert (
            o_c.is_experiment_name_in_fpath(
                "./data/campaign/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
            )
            is True
        )

        # INFO: Normal Case - with name in dir far from file
        assert (
            o_c.is_experiment_name_in_fpath(
                "./a/b/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/c/d/e/pub_0.csv"
            )
            is True
        )

        # INFO: Error Case - no name anywhere
        assert (
            o_c.is_experiment_name_in_fpath("./data/campaign/some_random_name.csv")
            is False
        )

    def test_recursively_get_fpaths(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - subdirs
        # There are 2 folders
        # Each folder has 84 files
        # 4 of these are csv files
        # There should be 84 x 2 = 168 files altogether
        ls_fpaths = o_c.recursively_get_fpaths(
            "./tests/data/test_campaign_with_dirs_small/"
        )
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 168

        for s_path in ls_fpaths:
            assert isinstance(s_path, str)
            assert os.path.exists(s_path)
            assert os.path.isfile(s_path)

        # INFO: Normal Case - no subdirs
        ls_fpaths = o_c.recursively_get_fpaths("./tests/data/test_campaign_with_csv/")
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 4

        # INFO: Normal Case - sub dirs and sub sub dirs
        # 3 files in /
        # 2 files in /logs
        # 1 file in /logs/test_dir
        # total should be 3 + 2 + 1
        ls_fpaths = o_c.recursively_get_fpaths(
            "./tests/data/test_campaign_with_mix/600s_100B_1P_1S_be_uc_3dur_100lc/"
        )
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 6
