import pytest
import os
import pandas as pd

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

    def test_create_dataset(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS

        for d_ds in ld_ds_config:
            o_c = Campaign(d_ds)
            s_raw_datadir = o_c.get_raw_datadir()
            ld_exp_names_and_paths = o_c.get_experiments(s_raw_datadir)

            # INFO: Normal case - should create dataset
            o_c.create_dataset()

            assert os.path.exists(o_c.ds_output_path)
            assert os.path.isfile(o_c.ds_output_path)
            assert os.path.getsize(o_c.ds_output_path) > 0
            assert o_c.ds_df is not None
            assert isinstance(o_c.ds_df, pd.DataFrame)
            assert len(o_c.ds_df) > 0
            assert len(o_c.ds_df.columns) > 0

            ls_wanted_cols = [
                'experiment_name',
                "latency_us",
                'avg_mbps',
                'total_mbps',
            ]
            for s_col in ls_wanted_cols:
                assert s_col in o_c.ds_df.columns
                assert o_c.ds_df[s_col].dtype == 'float64'

            df = o_c.get_ds_df()
            ls_exp_names = df['experiment_name'].unique().tolist()

            for d_exp in ld_exp_names_and_paths:
                s_exp_name = d_exp['name']
                assert s_exp_name in ls_exp_names

                df_exp = df[df['experiment_name'] == s_exp_name].copy()
                assert len(df_exp) > 0
                assert len(df_exp.columns) > 0
                assert df_exp['experiment_name'].nunique() == 1
                assert df_exp['experiment_name'].iloc[0] == s_exp_name

                # INFO: Check latency_us
                df_exp_lat = df_exp[['latency_us']].copy()
                df_exp_lat.dropna(inplace=True)
                assert len(df_exp_lat) > 0

                # INFO: Check avg_mbps
                df_exp_avg_mbps = df_exp[['avg_mbps']].copy()
                df_exp_avg_mbps.dropna(inplace=True)
                # INFO: Check there are around 400 to 800 samples
                # This covers the expected 600 samples range
                assert len(df_exp_avg_mbps) > 400
                assert len(df_exp_avg_mbps) < 800

                # INFO: Check total_mbps
                df_exp_total_mbps = df_exp[['total_mbps']].copy()
                df_exp_total_mbps.dropna(inplace=True)
                # INFO: Check there are around 400 to 800 samples
                # This covers the expected 600 samples range
                assert len(df_exp_total_mbps) > 400
                assert len(df_exp_total_mbps) < 800

            
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

    def test_get_experiment_name_from_fpath(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS
        d_ds = ld_ds_config[0]
        o_c = Campaign(d_ds)

        # INFO: Name in file
        s_exp_name = o_c.get_experiment_name_from_fpath(
            "./data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        )

        assert isinstance(s_exp_name, str)
        assert len(s_exp_name) > 0
        assert s_exp_name == "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"

        # INFO: Name in directory
        s_exp_name = o_c.get_experiment_name_from_fpath(
            "./test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC/pub_0.csv"
        )

        assert isinstance(s_exp_name, str)
        assert len(s_exp_name) > 0
        assert s_exp_name == "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"

        # INFO: No name
        with pytest.raises(ValueError):
            s_exp_name = o_c.get_experiment_name_from_fpath(
                "./test_campaign_with_csv/pub_0.csv"
            )
            
    def test_follows_experiment_name_format(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS
        d_ds = ld_ds_config[0]
        o_c = Campaign(d_ds)

        ls_normal_cases = [
            # INFO: Normal case - most up to date format
            "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
            # INFO: P as PUB
            "600SEC_100B_15P_15SUB_BE_MC_3DUR_100LC",
            # INFO: S as SUB
            "600SEC_100B_15PUB_15S_BE_MC_3DUR_100LC",
            # INFO: P as PUB and S as SUB
            "600S_100B_15P_15S_BE_MC_3DUR_100LC",
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
            b_follows_format = o_c.follows_experiment_name_format(
                s_exp_name
            )
            assert isinstance(b_follows_format, bool)
            assert b_follows_format is True

    def test_get_experiment_paths_from_fpath(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS
        d_ds = ld_ds_config[0]
        o_c = Campaign(d_ds)

        # INFO: Normal Case
        ls_exp_paths = o_c.get_experiment_paths_from_fpath(
            "./tests/data/test_campaign_with_dirs_small/"
        )
        assert isinstance(ls_exp_paths, list)
        assert len(ls_exp_paths) == 14

        for s_path in ls_exp_paths:
            assert isinstance(s_path, str)
            assert os.path.exists(s_path)
            assert os.path.isfile(s_path)
            assert os.path.getsize(s_path) > 0
            assert s_path.endswith('.csv')
        
    def test_recursively_get_fpaths(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        ld_ds_config = LD_DATASETS
        d_ds = ld_ds_config[0]
        o_c = Campaign(d_ds)

        # INFO: Normal Case - subdirs
        ls_fpaths = o_c.recursively_get_fpaths(
            "./tests/data/test_campaign_with_dirs_small/"
        )
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 174

        for s_path in ls_fpaths:
            assert isinstance(s_path, str)
            assert os.path.exists(s_path)
            assert os.path.isfile(s_path)

        # INFO: Normal Case - no subdirs
        ls_fpaths = o_c.recursively_get_fpaths(
            "./tests/data/test_campaign_with_csv/"
        )
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 5

        # INFO: Normal Case - sub dirs and sub sub dirs
        ls_fpaths = o_c.recursively_get_fpaths(
            "./tests/data/test_campaign_with_mix/600s_100B_1P_1S_be_uc_3dur_100lc/"
        )
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 84
