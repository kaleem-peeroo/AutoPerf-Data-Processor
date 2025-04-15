import pytest
import os

class TestCampaign:
    def test_init_with_normal_case(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS

        for d_ds in ld_ds_config:
            o_c = Campaign(d_ds)
            assert o_c is not None
            assert isinstance(o_c, Campaign)

            assert o_c.ds_output_path == d_ds['dataset_path']
            assert o_c.apconf_path == d_ds['ap_config']
            assert o_c.raw_datadir == d_ds['exp_folders']
            assert o_c.ds_df is None
            assert o_c.config is None
            assert o_c.qos_variations == {}
            assert o_c.qos_exp_list == []
            assert o_c.data_exp_list == []
            assert o_c.missing_exps == []

    def test_get_experiments_with_normal_case(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS

        for d_ds in ld_ds_config:
            o_c = Campaign(d_ds)
            s_raw_datadir = o_c.get_raw_datadir()
            ld_exp_names_and_paths = o_c.get_experiments(s_raw_datadir)

            """
            Checks:
                - should be a list of dicts
                - each dict should have two items
                    - name: name of experiment
                        - should be a string
                        - format of {int}SEC_{int}B_{int}P_{int}S_{REL/BE}_{MC/UC}_{int}DUR
                    - paths: list of all files for that experiment
                        - should be a list of strings
                        - should be a valid path to file
            """

            assert isinstance(ld_exp_names_and_paths, list)
            assert len(ld_exp_names_and_paths) > 0
            for d_exp in ld_exp_names_and_paths:
                assert isinstance(d_exp, dict)
                assert len(d_exp) == 2
                assert 'name' in d_exp
                assert 'paths' in d_exp

                assert isinstance(d_exp['name'], str)
                assert len(d_exp['name']) > 0

                assert isinstance(d_exp['paths'], list)
                assert len(d_exp['paths']) > 0

                for s_path in d_exp['paths']:
                    assert isinstance(s_path, str)
                    assert len(s_path) > 0
                    assert os.path.exists(s_path)
                    assert os.path.isfile(s_path)
                    assert os.path.getsize(s_path) > 0
                    assert s_path.endswith('.csv')
