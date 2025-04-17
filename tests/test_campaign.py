import pytest
import os
import pandas as pd

from rich.pretty import pprint

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

    def test_process_exp_df_with_exp_name_as_csv(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        d_ds_config = LD_DATASETS[0]
        o_c = Campaign(d_ds_config)

        d_test_exp_names_and_paths = {
            'name': "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
            'paths': [
                './tests/data/test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv' 
            ]
        }

        df = o_c.process_exp_df(d_test_exp_names_and_paths)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

        ls_wanted_cols = [
            'experiment_name',
            "latency_us",
            'avg_mbps_per_sub',
            'total_mbps_over_subs',

            'duration_secs',
            'datalen_bytes',
            'pub_count',
            'sub_count',
            'use_reliable',
            'use_multicast',
            'durability'
        ]
        for s_col in ls_wanted_cols:
            assert s_col in df.columns

        df_lat = df[['latency_us']].copy().dropna()
        assert len(df_lat) == 5284

        df_avg_tp = df[['avg_mbps_per_sub']].copy().dropna()
        assert len(df_avg_tp) == 612

        df_total_tp = df[['total_mbps_over_subs']].copy().dropna()
        assert len(df_total_tp) == 5284


    def test_process_exp_df_with_raw_files(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        d_ds_config = LD_DATASETS[0]
        o_c = Campaign(d_ds_config)

        s_test_dir = "./tests/data/test_campaign_with_dirs_simple/"
        d_test_exp_names_and_paths = {
            'name': "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
            'paths': [
                f'{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv' ,
                f'{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_1.csv' 
            ]
        }

        df = o_c.process_exp_df(d_test_exp_names_and_paths)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

        ls_wanted_cols = [
            'experiment_name',
            "latency_us",
            'avg_mbps_per_sub',
            'total_mbps_over_subs',

            'duration_secs',
            'datalen_bytes',
            'pub_count',
            'sub_count',
            'use_reliable',
            'use_multicast',
            'durability'
        ]
        for s_col in ls_wanted_cols:
            assert s_col in df.columns

        df_lat = df[['latency_us']].copy().dropna()
        assert len(df_lat) == 5284

        df_avg_tp = df[['avg_mbps_per_sub']].copy().dropna()
        assert len(df_avg_tp) == 612

        df_total_tp = df[['total_mbps_over_subs']].copy().dropna()
        assert len(df_total_tp) == 5284

    def setup_test_calculate_averages(self):
        from app import Campaign

        d_conf = {
            "name": "test campaign with dirs",
            "exp_folders": \
                "./tests/data/test_campaign_with_dirs_small/",
            "ap_config": "",
            "dataset_path": \
                "./tests/output/test_campaign_with_dirs_small.parquet",
        }
        o_c = Campaign(d_conf)

        s_camp_dir = "./tests/data/test_campaign_with_dirs_small/"
        s_exp_dir = f"{s_camp_dir}300SEC_1B_1P_3S_BE_MC_0DUR_100LC/"
        d_exp_names_and_paths = {
            "name": "300SEC_1B_1P_3S_BE_MC_0DUR_100LC",
            "paths": [
                f"{s_exp_dir}pub_0.csv",
                f"{s_exp_dir}sub_0.csv",
                f"{s_exp_dir}sub_1.csv",
                f"{s_exp_dir}sub_2.csv",
                f"{s_exp_dir}sub_3.csv",
                f"{s_exp_dir}sub_4.csv",
                f"{s_exp_dir}sub_5.csv",
            ]
        }

        df_exp = pd.DataFrame()
        for s_exp_path in d_exp_names_and_paths['paths']:
            assert os.path.exists(s_exp_path)
            df_temp = o_c.get_exp_file_df(s_exp_path)

            if df_exp.empty:
                df_exp = df_temp
            else:
                df_exp = pd.concat([df_exp, df_temp], axis=1)

        df_exp['experiment_name'] = d_exp_names_and_paths['name']
        df_exp = o_c.add_input_cols(df_exp)

        ls_wanted_cols = [
            'sub_0_mbps',
            'sub_1_mbps',
            'sub_2_mbps',
            'sub_3_mbps',
            'sub_4_mbps',
            'sub_5_mbps',
        ]
        for s_col in ls_wanted_cols:
            assert s_col in df_exp.columns
        
        return o_c, df_exp

    def test_calculate_averages_for_avg_mbps_per_sub(self):
        o_c, df_before = self.setup_test_calculate_averages()
        
        df_after = o_c.calculate_averages_for_avg_mbps_per_sub(df_before.copy())

        assert df_after is not None
        assert isinstance(df_after, pd.DataFrame)
        assert len(df_after) > 0
        assert len(df_after.columns) == len(df_before.columns) + 1
        assert 'avg_mbps_per_sub' in df_after.columns

    def test_calculate_averages_for_total_mbps_over_subs(self):
        raise NotImplementedError("Test not implemented yet")

    def test_get_exp_file_df(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        d_ds_config = LD_DATASETS[0]
        o_c = Campaign(d_ds_config)

        # INFO: Normal Case - raw sub file
        s_raw_file = "./tests/data/test_campaign_with_dirs_small/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_2.csv"
        df = o_c.get_exp_file_df(s_raw_file)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        ls_expected_cols = [
            'sub_2_mbps',
        ]
        for s_col in ls_expected_cols:
            assert s_col in df.columns
            assert df[s_col].dtype == 'float64'

        # INFO: Normal Case - raw pub file
        s_raw_file = "./tests/data/test_campaign_with_dirs_small/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        df = o_c.get_exp_file_df(s_raw_file)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'latency_us' in df.columns
        assert df['latency_us'].dtype == 'float64'

    def test_is_raw_exp_file(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - pub file
        assert o_c.is_raw_exp_file(
            "./path/to/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        ) is True

        # INFO: Normal Case - sub file
        assert o_c.is_raw_exp_file(
            "./path/to/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_0.csv"
        ) is True

        # INFO: Normal Case - not raw file
        assert o_c.is_raw_exp_file(
            "./path/to/300SEC_1B_1P_3S_BE_MC_0DUR_100LC.csv"
        ) is False

        # INFO: Error Case - some random csv file
        with pytest.raises(ValueError):
            o_c.is_raw_exp_file(
                "./path/to/idk_what_this_file_is.csv"
            )

    def test_process_file_df(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS
        o_c = Campaign(LD_DATASETS[0])

        s_test_datadir = "./tests/data/test_campaign_with_dirs_simple/"

        # INFO: Normal Case - pub file
        df_pub = o_c.process_file_df(
            f"{s_test_datadir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        )
        assert df_pub is not None
        assert isinstance(df_pub, pd.DataFrame)
        assert len(df_pub) > 0
        assert "latency_us" in df_pub.columns
        assert df_pub["latency_us"].dtype == "float64"
        assert len(df_pub) == 11

        # INFO: Normal Case - sub file
        df_sub = o_c.process_file_df(
            f"{s_test_datadir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_1.csv"
        )
        assert df_sub is not None
        assert isinstance(df_sub, pd.DataFrame)
        assert len(df_sub) > 0
        assert "sub_1_mbps" in df_sub.columns
        assert df_sub["sub_1_mbps"].dtype == "float64"
        assert len(df_sub) == 298

    def test_get_metric_col_from_df(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - Latency (us)
        assert o_c.get_metric_col_from_df(
            pd.DataFrame({
                'Latency (us)': [1, 2, 3],
                'avg_mbps': [4, 5, 6],
                'total_mbps': [7, 8, 9],
            }),
            'latency'
        ) == "Latency (us)"

        # INFO: Normal Case - Mbps
        assert o_c.get_metric_col_from_df(
            pd.DataFrame({
                'Latency (us)': [1, 2, 3],
                'avg_mbps': [4, 5, 6],
                'mbps': [7, 8, 9],
            }),
            'mbps'
        ) == "mbps"

    def test_rename_df_col(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - Latency (us)
        df_before = pd.DataFrame({
            'Latency (us)': [1, 2, 3],
            'avg_mbps': [4, 5, 6],
            'total_mbps': [7, 8, 9],
        })
        df_after = o_c.rename_df_col(
            df_before.copy(),
            'Latency (us)',
            'latency_us'
        )
        assert df_after is not None
        assert isinstance(df_after, pd.DataFrame)
        assert len(df_after) == 3
        assert len(df_after.columns) == 3
        assert 'latency_us' in df_after.columns

        # INFO: Error Case - col not found
        with pytest.raises(ValueError):
            df_after = o_c.rename_df_col(
                df_before.copy(),
                'invalid_col',
                'latency_us'
            )

    def test_raw_file_is_pub(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - pub file
        assert o_c.raw_file_is_pub(
            "./path/to/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        ) is True

        # INFO: Normal Case - sub file
        assert o_c.raw_file_is_pub(
            "./path/to/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_1.csv"
        ) is False

        # INFO: Error case - neither pub or sub
        with pytest.raises(ValueError):
            o_c.raw_file_is_pub(
                "./path/to/300SEC_1B_1P_3S_BE_MC_0DUR_100LC.csv"
            )

    def test_get_start_index_for_raw_file(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        o_c = Campaign(LD_DATASETS[0])
        s_test_dir = "./tests/data/test_campaign_with_dirs_simple/"

        # INFO: Normal Case - pub file
        s_test_file = f"{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        i_start = o_c.get_start_index_for_raw_file(s_test_file)
        assert isinstance(i_start, int)
        assert i_start == 2

        # INFO: Normal Case - sub file
        s_test_file = f"{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_1.csv"
        i_start = o_c.get_start_index_for_raw_file(s_test_file)
        assert isinstance(i_start, int)
        assert i_start == 2

    def test_get_end_index_for_raw_file(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        o_c = Campaign(LD_DATASETS[0])
        s_test_dir = "./tests/data/test_campaign_with_dirs_simple/"

        # INFO: Normal Case - pub file
        s_test_file = f"{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        i_end = o_c.get_end_index_for_raw_file(s_test_file)
        assert isinstance(i_end, int)
        assert i_end == 13

        # INFO: Normal Case - sub file
        s_test_file = f"{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_1.csv"
        i_end = o_c.get_end_index_for_raw_file(s_test_file)
        assert isinstance(i_end, int)
        assert i_end == 300

    def test_add_input_cols(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        d_ds_config = LD_DATASETS[0]
        o_c = Campaign(d_ds_config)

        ls_input_cols = [
            'duration_secs',
            'datalen_bytes',
            'pub_count',
            'sub_count',
            'use_reliable',
            'use_multicast',
            'durability',
            'latency_count'
        ]

        # INFO: Normal Case
        df_before = pd.DataFrame({
            'experiment_name': ["600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"] * 3,
            'latency_us': [1, 2, 3],
            'avg_mbps': [4, 5, 6],
            'total_mbps': [7, 8, 9],
        })
        df_after = o_c.add_input_cols(df_before.copy())

        assert df_after is not None
        assert isinstance(df_after, pd.DataFrame)
        assert len(df_after) == 3
        assert len(df_after.columns) == len(df_before.columns) + len(ls_input_cols)
        assert df_after['experiment_name'].nunique() == 1

        for s_col in ls_input_cols:
            assert s_col in df_after.columns
            assert df_after[s_col].dtype == 'float64'
            assert df_after[s_col].nunique() == 1

        assert df_after['duration_secs'].iloc[0] == 600
        assert df_after['datalen_bytes'].iloc[0] == 100
        assert df_after['pub_count'].iloc[0] == 15
        assert df_after['sub_count'].iloc[0] == 15
        assert df_after['use_reliable'].iloc[0] == 0
        assert df_after['use_multicast'].iloc[0] == 1
        assert df_after['durability'].iloc[0] == 3
        assert df_after['latency_count'].iloc[0] == 100

        # INFO: Error Case - No experiment name col
        with pytest.raises(ValueError):
            df_before = pd.DataFrame({
                'latency_us': [1, 2, 3],
                'avg_mbps': [4, 5, 6],
                'total_mbps': [7, 8, 9],
            })
            df_after = o_c.add_input_cols(df_before.copy())

        # INFO: Error Case - Diff experiment names
        with pytest.raises(ValueError):
            df_before = pd.DataFrame({
                'experiment_name': [
                    "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
                    "600SEC_100B_10PUB_15SUB_BE_MC_3DUR_100LC",
                    "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC",
                ],
                'latency_us': [1, 2, 3],
                'avg_mbps': [4, 5, 6],
                'total_mbps': [7, 8, 9],
            })
            df_after = o_c.add_input_cols(df_before.copy())

    def test_get_qos_from_exp_name_with_normal_case(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        o_c = Campaign(LD_DATASETS[0])

        d_qos = o_c.get_qos_from_exp_name(
            "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"
        )

        assert d_qos is not None
        assert isinstance(d_qos, dict)
        assert d_qos == {
            'duration_secs': 600,
            'datalen_bytes': 100,
            'pub_count': 15,
            'sub_count': 15,
            'use_reliable': 0,
            'use_multicast': 1,
            'durability': 3,
            'latency_count': 100,
        }

    def test_get_qos_from_exp_name_with_invalid_cases(self):
        from app import Campaign
        from tests.configs.normal import LD_DATASETS

        o_c = Campaign(LD_DATASETS[0])

        ls_invalid_exp_names = [
            "100B_15PUB_15SUB_BE_MC_3DUR_100LC",
            "600SEC_15PUB_15SUB_BE_MC_3DUR_100LC",
            "600SEC_100B_15SUB_BE_MC_3DUR_100LC",
            "600SEC_100B_15PUB_BE_MC_3DUR_100LC",
            "600SEC_100B_15PUB_15SUB_MC_3DUR_100LC",
            "600SEC_100B_15PUB_15SUB_BE_3DUR_100LC",
            "600SEC_100B_15PUB_15SUB_BE_MC_100LC",
        ]

        for s_exp_name in ls_invalid_exp_names:
            with pytest.raises(ValueError):
                o_c.get_qos_from_exp_name(s_exp_name)

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
