import os
import re
import logging
import pandas as pd

from itertools import islice
from rich.pretty import pprint

lg = logging.getLogger(__name__)

class ExperimentFile:
    def __init__(
        self, 
        s_exp_name: str = "",
        s_path: str = ""
    ):
        if s_path == "":
            raise ValueError(
                "Experiment file path must not be empty"
            )

        if not os.path.exists(s_path):
            raise ValueError(
                f"Experiment file path does not exist: {s_path}"
            )

        if not s_path.endswith(".csv"):
            raise ValueError(
                f"Experiment file path must be a .csv file: {s_path}"
            )

        if s_exp_name == "":
            raise ValueError(
                "Experiment name must not be empty"
            )

        if s_exp_name.lower() not in s_path.lower():
            raise ValueError(
                f"Experiment name must be in the file path: {s_path}"
            )

        self.s_exp_name = s_exp_name
        self.s_path = s_path

    def __str__(self):
        return f"Experiment: {self.s_exp_name}, File Path: {self.s_path}"

    def get_s_path(self):
        if self.s_path == "":
            raise ValueError(
                "Experiment file path must not be empty"
            )

        if not os.path.exists(self.s_path):
            raise ValueError(
                f"Experiment file path does not exist: {self.s_path}"
            )

        if not self.s_path.endswith(".csv"):
            raise ValueError(
                f"Experiment file path must be a .csv file: {self.s_path}"
            )

        return self.s_path

    def is_raw(self):
        s_path = self.get_s_path()
        s_file_name = os.path.basename(s_path)

        # Check if file contains *pub_0.csv or *sub_n.csv

        if re.search(r"pub_0\.csv|sub_\d+\.csv", s_file_name):
            return True

        else:
            return False
        
    def is_pub(self):
        s_path = self.get_s_path()
        s_file_name = os.path.basename(s_path)

        # Check if file contains *pub_0.csv
        if re.search(r"pub_0\.csv", s_file_name):
            return True

        else:
            return False

    def is_sub(self):
        s_path = self.get_s_path()
        s_file_name = os.path.basename(s_path)

        # Check if file contains *sub_n.csv
        if re.search(r"sub_\d+\.csv", s_file_name):
            return True

        else:
            return False

    def is_valid(self):
        df = self.get_df()

        if df is None:
            lg.warning(f"DataFrame is None: {self.s_path}")
            return False

        if df.empty:
            lg.warning(f"DataFrame is empty: {self.s_path}")
            return False

        i_expected_samples = self.get_expected_sample_count()
        i_lower_bound = i_expected_samples - 60
        i_upper_bound = i_expected_samples + 60

        if self.is_pub():
            # INFO: If empty
            if df.shape[0] < 2:
                lg.warning("{self.s_path} is empty")
                return False

        elif self.is_sub():

            if df.shape[0] < i_lower_bound:
                lg.warning(
                    f"{self.s_path} has {df.shape[0]} samples. "
                    f"Expected {i_lower_bound} - {i_upper_bound} samples. "
                )
                return False

            if df.shape[0] > i_upper_bound:
                lg.warning(
                    f"{self.s_path} has {df.shape[0]} samples. "
                    f"Expected {i_lower_bound} - {i_upper_bound} samples. "
                )
                return False

        else:
            # INFO: Validating experiment csv
            if df.shape[0] < 2:
                lg.warning(f"{self.s_path} is empty")
                return False

            ls_sub_cols = [
                col for col in df.columns if re.search(r"sub_\d+", col)
            ]
            if len(ls_sub_cols) == 0:
                lg.warning(f"{self.s_path} has no sub columns")
                return False

            for s_sub_col in ls_sub_cols:
                df_sub = df[s_sub_col]
                df_sub = self.remove_trailing_zeroes(df_sub)
                df_sub = df_sub.dropna()

                if df_sub.shape[0] < 2:
                    lg.warning(
                        f"{self.s_path} has {df_sub.shape[0]} samples. "
                        f"Expected {i_lower_bound} - {i_upper_bound} samples. "
                    )
                    return False

                if df_sub.shape[0] < i_lower_bound:
                    lg.warning(
                        f"{self.s_path} {s_sub_col} has {df_sub.shape[0]} samples. "
                        f"Expected {i_lower_bound} - {i_upper_bound} samples. "
                    )
                    return False

                if df_sub.shape[0] > i_upper_bound:
                    lg.warning(
                        f"{self.s_path} {s_sub_col} has {df_sub.shape[0]} samples. "
                        f"Expected {i_lower_bound} - {i_upper_bound} samples. "
                    )
                    return False

        return True

    def get_expected_sample_count(self):
        s_exp_name = self.s_exp_name

        s_duration_part = s_exp_name.split("_")[0]
        # Remove all non-numeric characters
        s_duration_part = re.sub(r"\D", "", s_duration_part)
        i_duration = int(s_duration_part)
        return i_duration
                            
    def get_df(self):
        s_path = self.get_s_path()

        if not self.is_raw():
            df = pd.read_csv(s_path)

        else:
            i_start = self.get_start_index()
            i_end = self.get_end_index()

            df = pd.read_csv(
                s_path,
                skiprows=i_start,
                nrows=i_end - i_start,
                on_bad_lines="skip",
            )

            if self.is_sub():
                df = self.remove_trailing_zeroes(df)

        if df.empty:
            raise ValueError(
                f"Experiment file is empty: {s_path}"
            )

        return df

    def get_start_index(self):
        s_path = self.get_s_path()

        if not self.is_raw():
            raise ValueError(
                f"Can only get start index for raw files: {s_path}"
            )

        if self.is_pub():
            with open(s_path, "r") as o_file:
                ls_first_5_lines = []
                for i in range(5):
                    line = o_file.readline()
                    if not line:
                        break
                    ls_first_5_lines.append(line)

            start_index = 0
            for i, line in enumerate(ls_first_5_lines):
                if "Length (Bytes)" in line:
                    start_index = i
                    break

            if start_index == 0:
                raise ValueError(
                    f"Could not find start index for raw file: {s_path}"
                )

            return start_index

        elif self.is_sub():

            i_file_line_count = sum(1 for _ in open(s_path))
            i_chunk_size = i_file_line_count // 10

            li_interval_lines = []
            with open(s_path, "rb") as o_file:
                for i_chunk, chunk in enumerate(iter(
                    lambda: tuple(islice(o_file, i_chunk_size)), ()
                )):
                    for i_line, s_line in enumerate(chunk):
                        i_line_count = i_chunk * i_chunk_size + i_line

                        if b"interval" in s_line.lower():
                            li_interval_lines.append(i_line_count)

            if len(li_interval_lines) == 0:
                raise ValueError(
                    f"Could not find start index for raw file: {s_path}"
                )

            i_last_interval = li_interval_lines[-1]

            return i_last_interval + 1

        else:
            raise ValueError(
                f"Can only get start index for raw files: {s_path}"
            )

    def get_end_index(self):
        s_path = self.get_s_path()

        if not self.is_raw():
            raise ValueError(
                f"Can only get end index for raw files: {s_path}"
            )

        if self.is_pub():
            i_file_line_count = sum(1 for _ in open(s_path))
            i_chunk_size = i_file_line_count // 10

            b_found = False
            end_index = 0
            with open(s_path, "rb") as o_file:
                for i_chunk, chunk in enumerate(iter(
                    lambda: tuple(islice(o_file, i_chunk_size)), ()
                )):
                    if b_found:
                        break

                    for i_line, s_line in enumerate(chunk):
                        i_line_count = i_chunk * i_chunk_size + i_line
                        if b"summary" in s_line.lower() and not b_found:
                            end_index = i_line_count
                            b_found = True
                            break

                        elif b"interval" in s_line.lower() and \
                                not b_found and \
                                i_line_count > 10:
                            end_index = i_line_count
                            b_found = True
                            break

            if end_index <= 0 and not b_found:
                raise ValueError(
                    f"Could not find end index for raw file: {s_path}"
                )

            return end_index - 2

        elif self.is_sub():
            i_file_line_count = sum(1 for _ in open(s_path))
            i_chunk_size = i_file_line_count // 10

            b_found = False
            end_index = 0
            with open(s_path, "rb") as o_file:
                for i_chunk, chunk in enumerate(iter(
                    lambda: tuple(islice(o_file, i_chunk_size)), ()
                )):
                    if b_found:
                        break

                    for i_line, s_line in enumerate(chunk):
                        i_line_count = i_chunk * i_chunk_size + i_line
                        if b"summary" in s_line.lower() and not b_found:
                            end_index = i_line_count
                            b_found = True
                            break

            if end_index <= 0 and not b_found:
                lg.warning(f"Couldn't find 'summary' in {s_path}. Using last line.")
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
