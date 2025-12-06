import os
import re
import logging
import pandas as pd

from itertools import islice
from rich.pretty import pprint

lg = logging.getLogger(__name__)


class ExperimentFile:
    def __init__(self, s_exp_name: str = "", s_path: str = ""):
        if s_path == "":
            raise ValueError("Experiment file path must not be empty")

        if not os.path.exists(s_path):
            raise ValueError(f"Experiment file path does not exist: {s_path}")

        if not s_path.endswith(".csv"):
            raise ValueError(f"Experiment file path must be a .csv file: {s_path}")

        if s_exp_name == "":
            raise ValueError("Experiment name must not be empty")

        if s_exp_name.lower() not in s_path.lower():
            raise ValueError(f"Experiment name must be in the file path: {s_path}")

        self.s_exp_name = s_exp_name
        self.s_path = s_path

    def __str__(self):
        return f"Experiment: {self.s_exp_name}, File Path: {self.s_path}"

    def get_s_path(self):
        if self.s_path == "":
            raise ValueError("Experiment file path must not be empty")

        if not os.path.exists(self.s_path):
            raise ValueError(f"Experiment file path does not exist: {self.s_path}")

        if not self.s_path.endswith(".csv"):
            raise ValueError(f"Experiment file path must be a .csv file: {self.s_path}")

        return self.s_path

    def is_raw(self):
        s_path = self.get_s_path()
        s_file_name = os.path.basename(s_path)

        # INFO: Check if file contains *pub_0*.csv or *sub_n*.csv

        if re.search(r"(?:pub_0|sub_\d+)(?:_output)?\.csv", s_file_name):
            return True

        else:
            return False

    def is_pub(self):
        s_path = self.get_s_path()
        s_file_name = os.path.basename(s_path)

        # Check if file contains *pub_0*.csv
        if re.search(r"(?:pub_0)(?:_output)?\.csv", s_file_name):
            return True

        else:
            return False

    def is_sub(self):
        s_path = self.get_s_path()
        s_file_name = os.path.basename(s_path)

        # Check if file contains *sub_n*.csv
        if re.search(r"(?:sub_\d+)(?:_output)?\.csv", s_file_name):
            return True

        else:
            return False

    def get_expected_sample_count(self):
        s_exp_name = self.s_exp_name

        s_duration_part = s_exp_name.split("_")[0]
        # Remove all non-numeric characters
        s_duration_part = re.sub(r"\D", "", s_duration_part)
        i_duration = int(s_duration_part)
        return i_duration

    def get_line_count(self):
        with open(self.s_path, "rb") as f:
            i_lines = sum(1 for _ in f)

        return i_lines

    def get_df(self):
        lg.debug(
            f"Getting DF for {os.path.basename(self.s_path)} ({self.get_line_count()} lines)"
        )

        if not self.is_raw():
            df = pd.read_csv(self.get_s_path())

        else:
            df = self.parse_raw_file()

        lg.debug(f"Got the DF ({df.shape})")
        return df

    def parse_raw_file(self) -> pd.DataFrame:
        lg.debug(f"Parsing raw file:\n\t{self.s_path}")

        i_start = self.get_start_index()
        i_end = self.get_end_index()

        if i_start > i_end:
            raise ValueError(
                f"Somehow the start ({i_start}) is after the end ({i_end})..."
            )

        lg.debug(f"Parsing raw file between {i_start} to {i_end}...")

        df = pd.read_csv(
            self.get_s_path(),
            skiprows=i_start,
            nrows=i_end - i_start,
            on_bad_lines="skip",
        )

        df = self.clean_df_col_names(df)

        return df

    def clean_df_col_names(self, df: pd.DataFrame) -> pd.DataFrame:
        lg.debug(f"Cleaning {len(df.columns)} columns: {list(sorted(df.columns))}")
        df.columns = df.columns.str.strip()

        ls_cols = list(sorted(df.columns))
        ls_wanted_cols = []

        ls_wanted_metrics = ["mbps", "latency", "samples/s", "lost samples"]
        for s_col in ls_cols:
            if "avg" in s_col.lower() or "ave" in s_col.lower():
                lg.debug(f"Skipping {s_col} because its average")
                continue

            for s_metric in ls_wanted_metrics:
                if s_metric in s_col.lower():
                    lg.debug(f"Adding {s_col}")
                    ls_wanted_cols.append(s_col)

        ls_wanted_cols = list(set(ls_wanted_cols))
        df = df[ls_wanted_cols]

        # INFO: Rename the metric columns
        for s_col in df.columns:
            if "latency" in s_col.lower():
                s_new_name = "latency_us"

            elif "mbps" in s_col.lower():
                s_new_name = "mbps"

            elif "lost samples (%)" in s_col.lower():
                s_new_name = "lost_samples_percent"

            elif "lost samples" in s_col.lower():
                s_new_name = "lost_samples"

            elif "samples/s" in s_col.lower():
                s_new_name = "sample_rate"

            df = df.rename(columns={s_col: s_new_name})

        if self.is_sub():
            s_sub_name = os.path.basename(self.s_path).replace(".csv", "_")
            df = df.add_prefix(s_sub_name)
            s_mbps_col = self.get_mbps_col(df)
            df[s_mbps_col] = self.remove_trailing_zeroes(df[s_mbps_col])

        return df

    def get_mbps_col(self, df: pd.DataFrame) -> str:
        """
        Get the mbps column name from the DataFrame.
        """
        s_mbps_col = [col for col in df.columns if "mbps" in col.lower()]
        if len(s_mbps_col) == 0:
            raise ValueError(f"Could not find mbps column in {self.s_path}")

        return s_mbps_col[0]

    def get_start_index(self):
        i_total_lines = self.get_line_count()

        i_min_lines = 50
        i_first_n_lines = i_min_lines if i_total_lines > i_min_lines else i_total_lines

        lg.debug(f"Searching first {i_first_n_lines} lines for the start...")

        if self.is_pub():
            with open(self.get_s_path(), "r") as o_file:
                ls_first_portion_of_lines = []

                for i in range(i_first_n_lines):
                    line = o_file.readline()
                    if not line:
                        break
                    ls_first_portion_of_lines.append(line)

            start_index = 0
            for i, line in enumerate(ls_first_portion_of_lines):
                if "length (bytes)" in line.lower() and "latency" in line.lower():
                    start_index = i
                    break

            return start_index

        elif self.is_sub():

            i_file_line_count = sum(1 for _ in open(self.get_s_path()))
            i_chunk_size = i_file_line_count // 10

            start_index = 0
            with open(self.get_s_path(), encoding="utf-8") as o_file:
                for i_line, s_line in enumerate(o_file, 1):
                    if "interval" in s_line.lower():
                        return i_line

            return 0

        else:
            raise ValueError(
                f"Can only get start index for raw files: {self.get_s_path()}"
            )

    def get_end_index(self):
        s_path = self.get_s_path()

        if not self.is_raw():
            raise ValueError(f"Can only get end index for raw files: {s_path}")

        if self.is_pub():
            i_file_line_count = sum(1 for _ in open(s_path))
            i_chunk_size = i_file_line_count // 10

            with open(s_path, encoding="utf-8") as f:
                for i_line, s_line in enumerate(reversed(f.readlines()[-50_000:]), 1):
                    if "latency summary" in s_line.lower():
                        return sum(1 for _ in open(s_path)) - i_line - 1

            return 0

        elif self.is_sub():
            i_file_line_count = sum(1 for _ in open(s_path))
            i_chunk_size = i_file_line_count // 10

            b_found = False
            end_index = 0
            with open(s_path, "rb") as o_file:
                for i_chunk, chunk in enumerate(
                    iter(lambda: tuple(islice(o_file, i_chunk_size)), ())
                ):
                    if b_found:
                        break

                    for i_line, s_line in enumerate(chunk):
                        i_line_count = i_chunk * i_chunk_size + i_line
                        if b"summary" in s_line.lower() and not b_found:
                            end_index = i_line_count
                            b_found = True
                            break

            if end_index <= 0 and not b_found:
                # lg.warning(f"Couldn't find 'summary' in {s_path}. Using last line.")
                end_index = i_file_line_count

            return end_index - 2

    def remove_trailing_zeroes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove trailing zeroes from the DataFrame.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()

        mask = df.ne(0).any(axis=1)
        last_nonzero_idx = mask[::-1].idxmax()
        df = df.loc[:last_nonzero_idx]

        return df
