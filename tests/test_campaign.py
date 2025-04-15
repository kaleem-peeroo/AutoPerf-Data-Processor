import pytest

class TestCampaign:
    def test_init_with_normal_case(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS

        for d_ds in ld_ds_config:
            campaign = Campaign(d_ds)
            assert campaign is not None
            assert isinstance(campaign, Campaign)
            assert campaign.ds_output_path == d_ds['dataset_path']
            assert campaign.apconf_path == d_ds['ap_config']
            assert campaign.raw_datadir == d_ds['exp_folders']
