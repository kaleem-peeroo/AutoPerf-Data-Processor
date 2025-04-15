import pytest

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
