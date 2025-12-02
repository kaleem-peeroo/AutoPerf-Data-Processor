import os
import shutil
import pytest


class TestMain:
    def test_main_with_n_runs(self):
        from main import main
        from campaign import Campaign
        from tests.configs.n_runs import LD_DATASETS

        # INFO: Delete if already exists before the test
        if os.path.exists("./tests/output/test_campaign_with_n_runs.parquet"):
            os.remove("./tests/output/test_campaign_with_n_runs.parquet")

        if os.path.exists("./tests/output/test_campaign_with_n_runs_summaries"):
            shutil.rmtree("./tests/output/test_campaign_with_n_runs_summaries")

        for i_ds, d_ds in enumerate(LD_DATASETS):
            campaign = Campaign(d_ds)
            campaign.summarise_experiments()
            campaign.create_dataset()

            df_ds = campaign.df_ds

            ls_columns = list(df_ds.columns)
            ls_columns = sorted(ls_columns)
            ls_wanted_cols = [
                "run_n",
                "experiment_name",
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
                    s_col in ls_columns
                ), f"{s_col} NOT found in dataset: {ls_columns}."

    def test_validate_config_with_normal_case(self):
        from main import validate_config
        from tests.configs.normal import LD_DATASETS

        assert validate_config(LD_DATASETS) == LD_DATASETS

    def test_validate_config_with_empty_dict(self):
        from main import validate_config

        with pytest.raises(ValueError):
            validate_config({})

    def test_validate_with_empty_list(self):
        from main import validate_config

        with pytest.raises(ValueError):
            validate_config([])

    def test_validate_config_with_invalid_type(self):
        from main import validate_config

        with pytest.raises(ValueError):
            validate_config("invalid_type")

    def test_validate_config_with_missing_key(self):
        from main import validate_config

        with pytest.raises(ValueError):
            validate_config([{"name": "test"}])

    def test_validate_config_with_invalid_key(self):
        from main import validate_config

        with pytest.raises(ValueError):
            validate_config(
                [
                    {
                        "name": "test",
                        "exp_folders": "value",
                        "ap_config": "",
                        "invalid_key": "",
                    }
                ]
            )

    def test_validate_config_with_invalid_value(self):
        from main import validate_config

        with pytest.raises(ValueError):
            validate_config(
                [
                    {
                        "name": "test",
                        "exp_folders": 123,
                        "ap_config": "",
                        "dataset_path": "",
                    }
                ]
            )

    def test_validate_config_with_invalid_dataset_path(self):
        from main import validate_config

        with pytest.raises(ValueError):
            validate_config(
                [
                    {
                        "name": "test",
                        "exp_folders": "",
                        "ap_config": "",
                        "dataset_path": 123,
                    }
                ]
            )

    def test_validate_config_with_invalid_ap_config(self):
        from main import validate_config

        assert validate_config(
            [
                {
                    "name": "test",
                    "exp_folders": "./tests/data/test_campaign_with_csv/",
                    "ap_config": "",
                    "dataset_path": "./tests/output/test_campaign.parquet",
                }
            ]
        ) == [
            {
                "name": "test",
                "exp_folders": "./tests/data/test_campaign_with_csv/",
                "ap_config": "",
                "dataset_path": "./tests/output/test_campaign.parquet",
            }
        ]

    def test_validate_config_with_existing_dataset_path(self):
        from main import validate_config

        assert validate_config(
            [
                {
                    "name": "test",
                    "exp_folders": "./tests/data/test_campaign_with_csv",
                    "ap_config": "",
                    "dataset_path": "./tests/output/existing_test_campaign.parquet",
                }
            ]
        ) == [
            {
                "name": "test",
                "exp_folders": "./tests/data/test_campaign_with_csv",
                "ap_config": "",
                "dataset_path": "./tests/output/existing_test_campaign.parquet",
            }
        ]
