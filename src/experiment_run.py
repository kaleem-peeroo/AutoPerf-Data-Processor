import logging
import os
import pandas as pd

from typing import List

from experiment_file import ExperimentFile

lg = logging.getLogger(__name__)


class ExperimentRun:
    def __init__(
        self,
        s_exp_name: str = "",
        s_run_name: str = "",
        ls_csv_paths: List[str] = [],
    ):
        if s_exp_name == "":
            raise ValueError("Experiment name must not be empty")

        if s_run_name == "":
            raise ValueError("Run name must not be empty")

        if len(ls_csv_paths) == 0:
            raise ValueError("Experiment csv paths must not be empty")

        self.s_exp_name = s_exp_name
        self.s_run_name = s_run_name
        self.lo_exp_files = [
            ExperimentFile(s_exp_name, s_csv_path) for s_csv_path in ls_csv_paths
        ]

    def __repr__(self):
        return f"{self.s_run_name} ({len(self.lo_exp_files)} Files)"

    def has_raw_data(self):
        """
        Check if the run has raw data.
        """
        for o_exp_file in self.lo_exp_files:
            if not o_exp_file.is_raw():
                # lg.warning(
                #     f"{o_exp_file.s_exp_name} is not raw"
                # )
                return False

        return True

    def summarise(self):
        df_summary = pd.DataFrame()

        for o_exp_file in self.lo_exp_files:
            df_file = o_exp_file.get_df()
            df_summary = pd.concat([df_summary, df_file], axis=1).reindex(
                index=range(max(len(df_summary), len(df_file)))
            )

        df_summary["run_n"] = self.s_run_name
        df_summary = self.calculate_sub_metrics(df_summary)

        return df_summary

    def calculate_sub_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate subscriber metrics.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a dataframe" f"It is a {type(df)}")

        if df.empty:
            raise ValueError("Dataframe is empty")

        lg.debug("Calculating avg and total mbps...")

        # Calculate avg mbps per sub
        ls_mbps_cols = [col for col in df.columns if "mbps" in col.lower()]
        if len(ls_mbps_cols) == 0:
            raise ValueError(f"No mbps columns found in dataframe: {df.columns}")

        df["avg_mbps_per_sub"] = df[ls_mbps_cols].mean(axis=1)
        lg.debug("Calculated average mbps")
        df["total_mbps_over_subs"] = df[ls_mbps_cols].sum(axis=1)
        lg.debug("Calculated total mbps")

        return df
