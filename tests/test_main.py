import pytest


class TestMain:
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
