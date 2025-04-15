import pytest


class TestMain:
    def test_validate_config_with_normal_case(self):
        from main import validate_config
        from tests.configs.normal import LD_DATASETS

        assert validate_config(LD_DATASETS) == True

        
