import pytest
import os
import pandas as pd

from rich.pretty import pprint

from app import Campaign
from tests.configs.normal import LD_DATASETS

class TestCampaign:
    @pytest.fixture
    def setup_method(self, method):
        """
        Fixture to set up the test method.
        This will be called before each test method.
        """
        print(f"\nSetting up for test: {method.__name__}")

        # Remove everything in ./tests/output except for existing_test_campaign.parquet
        s_output_dir = "./tests/output/"

        if os.path.exists(s_output_dir):

            for s_file in os.listdir(s_output_dir):
                s_file_path = os.path.join(s_output_dir, s_file)

                if s_file == "existing_test_campaign.parquet":
                    continue

                if os.path.isfile(s_file_path):
                    os.remove(s_file_path)

                elif os.path.isdir(s_file_path):
                    os.rmdir(s_file_path)

    def test_init_with_normal_case(self):
        for d_ds in LD_DATASETS:
            o_c = Campaign(d_ds)
            assert o_c is not None
            assert isinstance(o_c, Campaign)

            assert o_c.ds_output_path == d_ds['dataset_path']
            assert o_c.apconf_path == d_ds['ap_config']
            assert o_c.raw_datadir == d_ds['exp_folders']
            assert o_c.df_ds is None
            assert o_c.config is None
            assert o_c.qos_variations == {}
            assert o_c.qos_exp_list == []
            assert o_c.data_exp_list == []
            assert o_c.missing_exps == []

    def test_create_dataset_with_dirs(self):
        d_config = {
            "name": "test campaign with dirs",
            "exp_folders": \
                "./tests/data/test_campaign_with_dirs_small/",
            "ap_config": "",
            "dataset_path": \
                "./tests/output/test_campaign_with_dirs_small.parquet",
        }

        o_c = Campaign(d_config)
        s_raw_datadir = o_c.get_raw_datadir()
        ld_exp_names_and_paths = o_c.get_experiments(s_raw_datadir)

        o_c.create_dataset()

        assert os.path.exists(o_c.ds_output_path)
        assert os.path.isfile(o_c.ds_output_path)
        assert os.path.getsize(o_c.ds_output_path) > 0
        assert o_c.df_ds is not None
        assert isinstance(o_c.df_ds, pd.DataFrame)
        assert len(o_c.df_ds) > 0
        assert len(o_c.df_ds.columns) > 0

        ls_wanted_cols = [
            'experiment_name',
            "latency_us",
            'avg_mbps_per_sub',
            'total_mbps_over_subs',
        ]
        for s_col in ls_wanted_cols:
            assert s_col in o_c.df_ds.columns

        df = o_c.get_df_ds()
        ls_exp_names = df['experiment_name'].unique().tolist()

        for d_exp in ld_exp_names_and_paths:
            s_exp_name = o_c.try_format_experiment_name( d_exp['name'] )
            assert s_exp_name in ls_exp_names

            df_exp = df[df['experiment_name'] == s_exp_name].copy()
            assert len(df_exp) > 0
            assert len(df_exp.columns) > 0
            assert df_exp['experiment_name'].nunique() == 1
            assert df_exp['experiment_name'].iloc[0] == s_exp_name

            # INFO: No duplicate columns
            ls_cols = df_exp.columns.tolist()
            assert len(ls_cols) == len(set(ls_cols))

            # INFO: Check latency_us
            df_exp_lat = df_exp[['latency_us']].copy()
            df_exp_lat.dropna(inplace=True)
            assert len(df_exp_lat) > 0

            # INFO: Check avg_mbps
            df_exp_avg_mbps = df_exp[['avg_mbps_per_sub']].copy()
            df_exp_avg_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_avg_mbps) > 400
            assert len(df_exp_avg_mbps) < 800

            # INFO: Check total_mbps
            df_exp_total_mbps = df_exp[['total_mbps_over_subs']].copy()
            df_exp_total_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_total_mbps) > 400
            assert len(df_exp_total_mbps) < 800

    def test_create_dataset_with_csv(self):
        d_config = {
            "name": "test campaign with csv",
            "exp_folders": \
                "./tests/data/test_campaign_with_csv/",
            "ap_config": "",
            "dataset_path": \
                "./tests/output/test_campaign_with_csv_small.parquet",
        }

        o_c = Campaign(d_config)
        s_raw_datadir = o_c.get_raw_datadir()
        ld_exp_names_and_paths = o_c.get_experiments(s_raw_datadir)

        o_c.create_dataset()

        assert os.path.exists(o_c.ds_output_path)
        assert os.path.isfile(o_c.ds_output_path)
        assert os.path.getsize(o_c.ds_output_path) > 0
        assert o_c.df_ds is not None
        assert isinstance(o_c.df_ds, pd.DataFrame)
        assert len(o_c.df_ds) > 0
        assert len(o_c.df_ds.columns) > 0

        ls_wanted_cols = [
            'experiment_name',
            "latency_us",
            'avg_mbps_per_sub',
            'total_mbps_over_subs',
        ]
        for s_col in ls_wanted_cols:
            assert s_col in o_c.df_ds.columns

        df = o_c.get_df_ds()
        ls_exp_names = df['experiment_name'].unique().tolist()

        for d_exp in ld_exp_names_and_paths:
            s_exp_name = d_exp['name']
            pprint(s_exp_name)
            assert s_exp_name in ls_exp_names

            df_exp = df[df['experiment_name'] == s_exp_name].copy()
            assert len(df_exp) > 0
            assert len(df_exp.columns) > 0
            assert df_exp['experiment_name'].nunique() == 1
            assert df_exp['experiment_name'].iloc[0] == s_exp_name

            # INFO: No duplicate columns
            ls_cols = df_exp.columns.tolist()
            assert len(ls_cols) == len(set(ls_cols))

            # INFO: Check latency_us
            df_exp_lat = df_exp[['latency_us']].copy()
            df_exp_lat.dropna(inplace=True)
            assert len(df_exp_lat) > 0

            # INFO: Check avg_mbps
            df_exp_avg_mbps = df_exp[['avg_mbps_per_sub']].copy()
            df_exp_avg_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_avg_mbps) > 400
            assert len(df_exp_avg_mbps) < 800

            # INFO: Check total_mbps
            df_exp_total_mbps = df_exp[['total_mbps_over_subs']].copy()
            df_exp_total_mbps.dropna(inplace=True)
            # INFO: Check there are around 400 to 800 samples
            # This covers the expected 600 samples range
            assert len(df_exp_total_mbps) > 400
            assert len(df_exp_total_mbps) < 800

    def test_process_exp_df_with_exp_name_as_csv(self):
        o_c = Campaign(LD_DATASETS[0])

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

        # INFO: Check for duplicate columns
        ls_cols = df.columns.tolist()
        assert len(ls_cols) == len(set(ls_cols))

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
        assert len(df_lat) == 784

        df_avg_tp = df[['avg_mbps_per_sub']].copy().dropna()
        assert len(df_avg_tp) == 612

        df_total_tp = df[['total_mbps_over_subs']].copy().dropna()
        assert len(df_total_tp) == 784

    def test_process_exp_df_with_raw_files(self):
        o_c = Campaign(LD_DATASETS[0])

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
        assert len(df_lat) == 251

        df_avg_tp = df[['avg_mbps_per_sub']].copy().dropna()
        assert len(df_avg_tp) == 563

        df_total_tp = df[['total_mbps_over_subs']].copy().dropna()
        assert len(df_total_tp) == 563

    def setup_test_calculate_metrics_for_subs(self):
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
        ]
        for s_col in ls_wanted_cols:
            assert s_col in df_exp.columns
        
        return o_c, df_exp

    def test_calculate_avg_mbps_per_sub_with_normal_case(self):
        o_c, df_before = self.setup_test_calculate_metrics_for_subs()
        
        df_after = o_c.calculate_avg_mbps_per_sub(df_before.copy())

        assert df_after is not None
        assert isinstance(df_after, pd.DataFrame)
        assert len(df_after) > 0
        assert len(df_after.columns) == len(df_before.columns) + 1
        assert 'avg_mbps_per_sub' in df_after.columns

    def test_calculate_avg_mbps_per_sub_with_already_existing_col(self):
        o_c, df_before = self.setup_test_calculate_metrics_for_subs()
        df_before['avg_mbps_per_sub'] = 0.0
        
        df_after = o_c.calculate_avg_mbps_per_sub(df_before.copy())

        assert df_after is not None
        assert isinstance(df_after, pd.DataFrame)
        assert len(df_after) > 0
        # INFO: Because it already exists, it should not add a new column
        assert len(df_after.columns) == len(df_before.columns)
        assert 'avg_mbps_per_sub' in df_after.columns

        # INFO: Check for duplicate cols
        ls_cols = df_after.columns.tolist()
        assert len(ls_cols) == len(set(ls_cols))

    def test_calculate_total_mbps_over_subs_with_normal_case(self):
        o_c, df_before = self.setup_test_calculate_metrics_for_subs()
        
        df_after = o_c.calculate_total_mbps_over_subs(df_before.copy())

        assert df_after is not None
        assert isinstance(df_after, pd.DataFrame)
        assert len(df_after) > 0
        assert len(df_after.columns) == len(df_before.columns) + 1
        assert 'total_mbps_over_subs' in df_after.columns

    def test_calculate_total_mbps_over_subs_with_already_existing_col(self):
        o_c, df_before = self.setup_test_calculate_metrics_for_subs()
        df_before['total_mbps_over_subs'] = 0.0
        
        df_after = o_c.calculate_total_mbps_over_subs(df_before.copy())

        assert df_after is not None
        assert isinstance(df_after, pd.DataFrame)
        assert len(df_after) > 0
        # INFO: Because it already exists, it should not add a new column
        assert len(df_after.columns) == len(df_before.columns)
        assert 'total_mbps_over_subs' in df_after.columns

        # INFO: Check for duplicate cols
        ls_cols = df_after.columns.tolist()
        assert len(ls_cols) == len(set(ls_cols))

    def test_get_sub_mbps_cols(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with 1 sub mbps col
        df = pd.DataFrame({
            'sub_0_mbps': [1, 2, 3],
            'latency_us': [7, 8, 9],
        })
        assert o_c.get_sub_mbps_cols(df) == ['sub_0_mbps']

        # INFO: Normal Case - with 2 sub mbps cols
        df = pd.DataFrame({
            'sub_0_mbps': [1, 2, 3],
            'sub_1_mbps': [4, 5, 6],
            'latency_us': [7, 8, 9],
        })
        assert o_c.get_sub_mbps_cols(df) == ['sub_0_mbps', 'sub_1_mbps']

        # INFO: Empty Case - no sub mbps cols
        df = pd.DataFrame({
            'latency_us': [7, 8, 9],
        })
        assert o_c.get_sub_mbps_cols(df) == []
        
    def test_get_exp_file_df(self):
        o_c = Campaign(LD_DATASETS[0])

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

        # INFO: Normal Case - some random csv file
        o_c.is_raw_exp_file(
            "./path/to/idk_what_this_file_is.csv"
        ) is False

    def test_process_file_df(self):
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
        assert len(df_pub) == 251

        # INFO: Normal Case - sub file
        df_sub = o_c.process_file_df(
            f"{s_test_datadir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_1.csv"
        )
        assert df_sub is not None
        assert isinstance(df_sub, pd.DataFrame)
        assert len(df_sub) > 0
        assert "sub_1_mbps" in df_sub.columns
        assert df_sub["sub_1_mbps"].dtype == "float64"
        assert len(df_sub) == 563

        # INFO: Normal Case - erroneous sub file
        s_test_datadir = "./tests/data/test_campaign_with_errors/"
        df_sub = o_c.process_file_df(
            f"{s_test_datadir}/600SEC_100B_25P_1S_BE_MC_2DUR_100LC/sub_0.csv"
        )
        assert df_sub is not None
        assert isinstance(df_sub, pd.DataFrame)
        assert len(df_sub) > 0
        assert "sub_0_mbps" in df_sub.columns
        assert df_sub["sub_0_mbps"].dtype == "float64"
        assert len(df_sub) == 892

    def test_get_metric_col_from_df(self):
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
        o_c = Campaign(LD_DATASETS[0])
        s_test_dir = "./tests/data/test_campaign_with_dirs_simple/"

        # INFO: Normal Case - pub file
        s_test_file = f"{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        i_end = o_c.get_end_index_for_raw_file(s_test_file)
        assert isinstance(i_end, int)
        assert i_end == 253

        # INFO: Normal Case - sub file
        s_test_file = f"{s_test_dir}/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/sub_1.csv"
        i_end = o_c.get_end_index_for_raw_file(s_test_file)
        assert isinstance(i_end, int)
        assert i_end == 565

        # INFO: Error Case - pub file with summary in middle
        s_test_dir = "./tests/data/test_campaign_with_errors/"
        s_test_file = f"{s_test_dir}/300SEC_100B_1P_1S_REL_UC_0DUR_100LC/pub_0.csv"
        i_end = o_c.get_end_index_for_raw_file(s_test_file)
        assert isinstance(i_end, int)
        assert i_end == 65414

        # INFO: Error Case - sub file with summary in middle
        s_test_dir = "./tests/data/test_campaign_with_errors/"
        s_test_file = f"{s_test_dir}/600SEC_100B_25P_1S_BE_MC_2DUR_100LC/sub_0.csv"
        i_end = o_c.get_end_index_for_raw_file(s_test_file)
        assert isinstance(i_end, int)
        # "Interval" is seen on line 897
        assert i_end == 894

    def test_add_input_cols(self):
        o_c = Campaign(LD_DATASETS[0])

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
        for d_ds in LD_DATASETS:
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

    def test_get_experiments_with_csv(self):
        o_c = Campaign({
            "name": "another test campaign with csv",
            "exp_folders": \
                "./tests/data/test_campaign_with_dir_with_csv/",
            "ap_config": "",
            "dataset_path": \
                "./tests/output/test_campaign_with_dir_with_csv_small.parquet",
        })

        s_raw_datadir = o_c.get_raw_datadir()
        ld_exp_names_and_paths = o_c.get_experiments(s_raw_datadir)

        assert len(ld_exp_names_and_paths) == 18

        for d_exp_names_and_paths in ld_exp_names_and_paths:
            for s_path in d_exp_names_and_paths['paths']:
                assert isinstance(s_path, str)
                assert len(s_path) > 0
                assert s_path.endswith('.csv')
                assert os.path.exists(s_path)

    def test_process_exp_entries_with_subdirs(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with dirs
        s_test_dir = "./tests/data/test_campaign_with_dirs/"
        ls_exp_entries = os.listdir(s_test_dir)
        ls_exp_entries = [
            os.path.join(s_test_dir, s_exp_entry)
            for s_exp_entry in ls_exp_entries
        ]
        ls_exp_entries = o_c.process_exp_entries_with_subdirs(ls_exp_entries)
        # There are 109 experiment folders
        assert len(ls_exp_entries) == 81

        # INFO: Normal Case - with subdirs
        s_test_dir = "./tests/data/test_campaign_with_dir_with_csv/"
        ls_exp_entries = os.listdir(s_test_dir)
        ls_exp_entries = [
            os.path.join(s_test_dir, s_exp_entry)
            for s_exp_entry in ls_exp_entries
        ]
        ls_exp_entries = o_c.process_exp_entries_with_subdirs(ls_exp_entries)
        # There are 18 experiment csv files
        assert len(ls_exp_entries) == 18

        # INFO: Normal Case - with csv
        s_test_dir = "./tests/data/test_campaign_with_csv/"
        ls_exp_entries = os.listdir(s_test_dir)
        ls_exp_entries = [
            os.path.join(s_test_dir, s_exp_entry)
            for s_exp_entry in ls_exp_entries
        ]
        ls_exp_entries = o_c.process_exp_entries_with_subdirs(ls_exp_entries)
        # There are 4 experiment csv files
        assert len(ls_exp_entries) == 4

    def test_contains_dirs(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with dirs
        s_test_dir = "./tests/data/test_campaign_with_dirs/"
        assert o_c.contains_dirs(s_test_dir) is True

        # INFO: Normal Case - without dirs
        s_test_dir = "./tests/data/test_campaign_with_csv/"
        assert o_c.contains_dirs(s_test_dir) is False

    def test_check_for_expected_files(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with expected files
        ld_exp_names_and_paths = [{
            'name': "600SEC_100B_1PUB_2SUB_BE_MC_3DUR_100LC",
            'paths': [
                "pub_0.csv",
                "sub_0.csv",
                "sub_1.csv",
            ]
        }]
        assert o_c.get_exp_with_expected_file_count(
            ld_exp_names_and_paths
        ) == ld_exp_names_and_paths

        # INFO: Normal Case - old format with p and s
        ld_exp_names_and_paths = [{
            'name': "600SEC_100B_1P_2S_BE_MC_3DUR_100LC",
            'paths': [
                "pub_0.csv",
                "sub_0.csv",
                "sub_1.csv",
            ]
        }]
        assert o_c.get_exp_with_expected_file_count(
            ld_exp_names_and_paths
        ) == ld_exp_names_and_paths

        # INFO: Normal Case - 1 experiment csv
        ld_exp_names_and_paths = [{
            'name': "600SEC_100B_1PUB_2SUB_BE_MC_3DUR_100LC",
            'paths': [
                "600SEC_100B_1PUB_2SUB_BE_MC_3DUR_100LC.csv",
            ]
        }]
        assert o_c.get_exp_with_expected_file_count(
            ld_exp_names_and_paths
        ) == ld_exp_names_and_paths

        # INFO: Error Case - missing sub files
        ld_exp_names_and_paths = [{
            'name': "600SEC_100B_1PUB_2SUB_BE_MC_3DUR_100LC",
            'paths': [
                "pub_0.csv",
                "sub_0.csv",
            ]
        }]
        assert o_c.get_exp_with_expected_file_count(ld_exp_names_and_paths) == []

        # INFO: Error Case - missing pub file
        ld_exp_names_and_paths = [{
            'name': "600SEC_100B_1PUB_2SUB_BE_MC_3DUR_100LC",
            'paths': [
                "sub_0.csv",
                "sub_1.csv",
            ]
        }]
        assert o_c.get_exp_with_expected_file_count(ld_exp_names_and_paths) == []

    def test_get_expected_file_count(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - latest name format
        assert o_c.get_expected_file_count(
            "600SEC_100B_1PUB_2SUB_BE_MC_3DUR_100LC"
        ) == 3

        # INFO: Normal Case - old name format
        assert o_c.get_expected_file_count(
            "600SEC_100B_1p_2s_BE_MC_3DUR_100LC"
        ) == 3

        # INFO: Error Case - invalid name
        with pytest.raises(ValueError):
            o_c.get_expected_file_count(
                "invalid_experiment_name"
            )
    
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

    def test_is_exp_name_in_dirpath(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with name in dir
        assert o_c.is_exp_name_in_dirpath(
            "./test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC/pub_0.csv"
        ) is True

        # INFO: Normal Case - name in file
        assert o_c.is_exp_name_in_dirpath(
            "./test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        ) is False

        # INFO: Error Case - no name anywhere
        assert o_c.is_exp_name_in_dirpath(
            "./test_campaign_with_csv/some_random_name.csv"
        ) is False

    def test_is_exp_name_in_filename(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - with name in filename
        assert o_c.is_exp_name_in_filename(
            "./test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        ) is True

        # INFO: Normal Case - with name in dir
        assert o_c.is_exp_name_in_filename(
            "./test_campaign_with_csv/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC/pub_0.csv"
        ) is False

        # INFO: Error Case - no name anywhere
        assert o_c.is_exp_name_in_filename(
            "./test_campaign_with_csv/some_random_name.csv"
        ) is False

    def test_try_format_experiment_name_in_path(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - most up to date format
        s_exp_name = o_c.try_format_experiment_name_in_path(
            "some/path/to/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name == "some/path/to/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"

        # INFO: Normal Case - with .csv at end
        s_exp_name = o_c.try_format_experiment_name_in_path(
            "path/to/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name == "path/to/600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"

        # INFO: Normal Case - old format with p and s
        s_exp_name = o_c.try_format_experiment_name_in_path(
            "path/to/600s_32000B_5P_1S_rel_mc_2dur_100lc"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name.lower() == "path/to/600sec_32000b_5pub_1sub_rel_mc_2dur_100lc"

        # INFO: Normal Case - old format with p and s with file extension
        s_exp_name = o_c.try_format_experiment_name_in_path(
            "path/to/600s_32000B_5P_1S_rel_mc_2dur_100lc.csv"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name.lower() == "path/to/600sec_32000b_5pub_1sub_rel_mc_2dur_100lc.csv"

        # INFO: Normal Case - entire path with dirs
        s_exp_name = o_c.try_format_experiment_name_in_path(
            "path/to/600s_32000B_5P_1S_rel_mc_2dur_100lc/pub_0.csv"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name.lower() == \
                "path/to/600sec_32000b_5pub_1sub_rel_mc_2dur_100lc/pub_0.csv"

        # INFO: Error Case - invalid name
        with pytest.raises(ValueError):
            s_exp_name = o_c.try_format_experiment_name_in_path(
                "invalid_experiment_name"
            )

        # INFO: Error Case - valid name but not a path
        with pytest.raises(ValueError):
            s_exp_name = o_c.try_format_experiment_name_in_path(
                "600sec_32000b_5pub_1sub_rel_mc_2dur_100lc.csv"
            )

    def test_try_format_experiment_name(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - most up to date format
        s_exp_name = o_c.try_format_experiment_name(
            "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name == "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC"

        # INFO: Normal Case - with .csv at end
        s_exp_name = o_c.try_format_experiment_name(
            "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name == "600SEC_100B_15PUB_15SUB_BE_MC_3DUR_100LC.csv"

        # INFO: Normal Case - old format with p and s
        s_exp_name = o_c.try_format_experiment_name(
            "600s_32000B_5P_1S_rel_mc_2dur_100lc"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name.lower() == "600sec_32000b_5pub_1sub_rel_mc_2dur_100lc"

        # INFO: Normal Case - old format with p and s with file extension
        s_exp_name = o_c.try_format_experiment_name(
            "600s_32000B_5P_1S_rel_mc_2dur_100lc.csv"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name.lower() == "600sec_32000b_5pub_1sub_rel_mc_2dur_100lc.csv"

        # INFO: Error Case - invalid name
        s_exp_name = o_c.try_format_experiment_name(
            "invalid_experiment_name"
        )
        assert isinstance(s_exp_name, str)
        assert s_exp_name == "invalid_experiment_name"

    def test_is_path(self):
        o_c = Campaign(LD_DATASETS[0])

        # Normal Case - path to folder
        assert o_c.is_path(
            "./tests/data/test_campaign_with_dirs_small/"
        ) is True

        # Normal Case - path to file
        assert o_c.is_path(
            "./tests/data/test_campaign_with_dirs_small/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        ) is True

        # Normal Case - not path
        assert o_c.is_path(
            "file.csv"
        ) is False

        # Error Case - numbers?
        with pytest.raises(ValueError):
            o_c.is_path(
                123456
            )
            
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
            b_follows_format = o_c.follows_experiment_name_format(
                s_exp_name
            )
            assert isinstance(b_follows_format, bool)
            assert b_follows_format is True, f"Failed for {s_exp_name}"

        # INFO: Error Case - file path to valid file
        s_exp_name = "./tests/data/test_campaign_with_dirs_small/300SEC_1B_1P_3S_BE_MC_0DUR_100LC/pub_0.csv"
        assert o_c.follows_experiment_name_format(
            s_exp_name
        ) is False

    def test_get_experiment_paths_from_fpath(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case
        ls_exp_paths = o_c.get_experiment_paths_from_fpath(
            "./tests/data/test_campaign_with_dirs_small/"
        )
        assert isinstance(ls_exp_paths, list)
        assert len(ls_exp_paths) == 8

        for s_path in ls_exp_paths:
            assert isinstance(s_path, str)
            assert os.path.exists(s_path)
            assert os.path.isfile(s_path)
            assert os.path.getsize(s_path) > 0
            assert s_path.endswith('.csv')

        # INFO: Normal Case - with subdirs
        ls_exp_paths = o_c.get_experiment_paths_from_fpath(
            "./tests/data/test_campaign_with_dir_with_csv/"
        )
        assert isinstance(ls_exp_paths, list)
        assert len(ls_exp_paths) == 18

        for s_path in ls_exp_paths:
            assert isinstance(s_path, str)
            assert os.path.exists(s_path)
            assert os.path.isfile(s_path)
            assert os.path.getsize(s_path) > 0
            assert s_path.endswith('.csv')
        
    def test_recursively_get_fpaths(self):
        o_c = Campaign(LD_DATASETS[0])

        # INFO: Normal Case - subdirs
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
        ls_fpaths = o_c.recursively_get_fpaths(
            "./tests/data/test_campaign_with_csv/"
        )
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 4

        # INFO: Normal Case - sub dirs and sub sub dirs
        ls_fpaths = o_c.recursively_get_fpaths(
            "./tests/data/test_campaign_with_mix/600s_100B_1P_1S_be_uc_3dur_100lc/"
        )
        assert isinstance(ls_fpaths, list)
        assert len(ls_fpaths) == 84
