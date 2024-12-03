import os
import pandas as pd

from rich.pretty import pprint

from logger import logger

class Experiment:
    def __init__(self, name):
        self.name       = name
        self.sub_files  = []
        self.subs_df    = None
        self.pub_file   = None
        self.pub_df     = None
        self.qos        = self.get_qos_from_name()

    def __rich_repr__(self):
        yield "name", self.name
        yield "sub_files", self.sub_files
        yield "subs_df", self.subs_df
        yield "pub_file", self.pub_file
        yield "pub_df", self.pub_df
        yield "qos", self.qos

    def get_qos_from_name(self):
        duration_secs = 0
        datalen_bytes = 0
        pub_count = 0
        sub_count = 0
        use_reliable = False
        use_multicast = False
        durability = 0
        latency_count = 0

        if self.name == "":
            raise ValueError("experiment name must not be empty")

        if not isinstance(self.name, str):
            raise ValueError(f"experiment name must be a string: {self.name}")

        self.name = self.name.replace("PUB_", "P_")
        self.name = self.name.replace("SUB_", "S_")

        self.name_parts = self.name.split("_")
        if len(self.name_parts) != 8:
            raise ValueError("{} must have 8 parts but has {}".format(
                self.name, len(self.name_parts)
            ))

        for part in self.name_parts:
            if part == "":
                raise ValueError("experiment name part must not be empty")

            if part.endswith("SEC"):
                duration_secs = int(part[:-3])

            elif part.endswith("B"):
                datalen_bytes = int(part[:-1])

            elif part.endswith("LC"):
                latency_count = int(part[:-2])

            elif part.endswith("DUR"):
                durability = int(part[:-3])

            elif (part == "UC") or (part == "MC"):

                if part == "UC":
                    use_multicast = False
                else:
                    use_multicast = True

            elif (part == "REL") or (part == "BE"):

                if part == "REL":
                    use_reliable = True
                else:
                    use_reliable = False

            elif part.endswith("P"):
                pub_count = int(part[:-1])

            elif part.endswith("S"):
                sub_count = int(part[:-1])

            else:
                raise ValueError(f"Unknown experiment name part: {part}")

        qos = {
            "duration_secs": duration_secs,
            "datalen_bytes": datalen_bytes,
            "pub_count": pub_count,
            "sub_count": sub_count,
            "use_reliable": use_reliable,
            "use_multicast": use_multicast,
            "durability": durability,
            "latency_count": latency_count
        }

        return qos
        
    def get_qos(self):
        return self.qos
    
    def get_name(self):
        return self.name

    def get_dirpath(self):
        return self.dirpath

    def get_sub_files(self):
        return self.sub_files

    def get_subs_df(self):
        return self.subs_df

    def get_pub_file(self):
        return self.pub_file

    def get_pub_df(self):
        return self.pub_df

    def set_qos(self, qos):
        self.qos = qos

    def set_name(self, name):
        self.name = name

    def set_dirpath(self, dirpath):
        self.dirpath = dirpath

    def set_sub_files(self, sub_files):
        if not isinstance(sub_files, list):
            raise Exception("sub_files must be a list")

        for file in sub_files:
            if "sub_" not in file:
                raise Exception("sub_files must contain files with 'sub' in the name")

            if not file.endswith(".csv"):
                raise Exception("sub_files must contain .csv files")

            if not os.path.exists(file):
                raise Exception(f"File does not exist: {file}")

        self.sub_files = sub_files

    def set_pub_file(self, pub_file):
        if not pub_file.endswith(".csv"):
            raise Exception("pub_file must be a .csv file")

        if not os.path.exists(pub_file):
            raise Exception(f"File does not exist: {pub_file}")

        if not pub_file.endswith("pub_0.csv"):
            raise Exception("pub_file must end with 'pub_0.csv'")

        self.pub_file = pub_file

    def get_expected_file_count(self):
        if "SUB_" in self.name:
            sub_count = self.name.split("SUB_")[0].split("_")[-1]
            return int(sub_count) + 1

        elif "S_" in self.name:
            sub_count = self.name.split("S_")[0].split("_")[-1]
            return int(sub_count) + 1

    def validate_files(self):
        if not self.pub_file:
            raise Exception("No pub file")

        if not self.sub_files:
            raise Exception("No sub files")

        expected_file_count = self.get_expected_file_count()
        is_valid = True

        total_file_count = len(self.sub_files) + 1
        if total_file_count != expected_file_count:
            logger.warning(f"{self.name}: Expected {expected_file_count} files, got {total_file_count}")
            is_valid = False

        for file in self.sub_files:
            if not os.path.exists(file):
                logger.warning(f"{self.name}: File does not exist: {file}")
                is_valid = False

            if not file.endswith(".csv"):
                logger.warning(f"{self.name}: File does not end with .csv: {file}")
                is_valid = False

        if not is_valid:
            return False

        return True

    def parse_sub_files(self):
        if len(self.sub_files) == 0:
            raise Exception("No sub files")

        sub_files = self.sub_files
        subs_df = pd.DataFrame()

        for sub_file in sub_files:
            # ? Find out where to start parsing the file from
            with open(sub_file, "r") as file_obj:
                if os.stat(sub_file).st_size == 0:
                    continue
                file_obj.seek(0)
                sub_first_5_lines = [next(file_obj) for _ in range(5)]

            start_index = 0
            for i, line in enumerate(sub_first_5_lines):
                if "Length (Bytes)" in line:
                    start_index = i
                    break

            if start_index == 0:
                logger.warning(f"Couldn't get start_index for header row from {os.path.basename(sub_file)}.")
                return None

            # ? Find out where to stop parsing the file from (ignore the summary stats at the end)
            with open(sub_file, "r") as file_obj:
                file_contents = file_obj.readlines()
            sub_last_5_lines = file_contents[-5:]
            line_count = len(file_contents)

            end_index = 0
            for i, line in enumerate(sub_last_5_lines):
                if "throughput summary" in line.lower():
                    end_index = line_count - 5 + i - 2
                    break

            if end_index == 0:
                logger.warning("Couldn't get end_index for summary row. File writing might have been interrupted.")
                end_index = line_count - 1

            nrows = end_index - start_index
            nrows = 0 if nrows < 0 else nrows

            try:
                df = pd.read_csv(
                    sub_file, on_bad_lines="skip", skiprows=start_index, nrows=nrows
                )
            except pd.errors.ParserError as e:
                logger.warning(f"Error when getting data from {os.path.basename(sub_file)}:{e}")
                return None

            desired_metrics = ["total samples", "samples/s", "mbps", "lost samples"]

            sub_name = os.path.basename(sub_file).replace(".csv", "")

            for col in df.columns:
                for desired_metric in desired_metrics:
                    if desired_metric in col.lower() and "avg" not in col.lower():
                        col_name = col.strip().lower().replace(" ", "_")

                        if "samples/s" in col_name:
                            col_name = "samples_per_sec"
                        elif "%" in col_name:
                            col_name = "lost_samples_percent"

                        subs_df[f"{sub_name}_{col_name}"] = df[col]
                        subs_df[f"{sub_name}_{col_name}"] = subs_df[
                            f"{sub_name}_{col_name}"
                        ].astype(float, errors="ignore")

        if subs_df.empty:
            logger.warning("Subscriber data is empty")
            return None

        self.subs_df = subs_df

    def get_colname(self, coltype, colnames):
        if coltype == "":
            return "", "Col name type not specified"

        if len(colnames) == 0:
            return "", "Col names not specified"

        for colname in colnames:
            if coltype in colname.lower():
                return colname, None

        return "", f"Couldn't find {coltype} colname"

    def parse_pub_file(self):
        pub_file = self.pub_file
        if pub_file == "":
            return None, "Publisher file not specified"

        with open(pub_file, "r") as pub_file_obj:
            pub_first_5_lines = []
            for i in range(5):
                line = pub_file_obj.readline()
                if not line:
                    break
                pub_first_5_lines.append(line)

        start_index = 0
        for i, line in enumerate(pub_first_5_lines):
            if "Ave" in line and "Length (Bytes)" in line:
                start_index = i
                break

        if start_index == 0:
            logger.warning(f"Couldn't find start index for header row for {os.path.basename(pub_file)}.")
            return None

        # ? Find out where to stop parsing the file from (ignore the summary stats at the end)
        with open(pub_file, "r") as pub_file_obj:
            pub_file_contents = pub_file_obj.readlines()

        pub_last_5_lines = pub_file_contents[-5:]
        line_count = len(pub_file_contents)

        end_index = 0
        for i, line in enumerate(pub_last_5_lines):
            if "latency summary" in line.lower():
                end_index = line_count - 5 + i - 2
                break

        if end_index == 0:
            end_index = line_count - 1

        try:
            lat_df = pd.read_csv(
                pub_file,
                skiprows=start_index,
                nrows=end_index - start_index,
                on_bad_lines="skip",
            )
        except pd.errors.EmptyDataError:
            logger.warning("EmptyDataError")
            return None

        min_colname, error = self.get_colname("min", lat_df.columns)
        if error:
            logger.warning(f"Error getting min colname for {pub_file}: {error}")
            return None

        max_colname, error = self.get_colname("max", lat_df.columns)
        if error:
            logger.warning(f"Error getting max colname for {pub_file}: {error}")
            return None

        if len(lat_df) == 0:
            logger.warning("Publisher data is empty")
            return None

        first_row = lat_df.iloc[0]
        first_min = first_row[min_colname]
        first_max = first_row[max_colname]

        first_latency_values = list(set([first_min, first_max]))

        # ? Pick out the latency column ONLY
        latency_col = None
        for col in lat_df.columns:
            if "latency" in col.lower():
                latency_col = col
                break

        if latency_col is None:
            logger.warning("Couldn't find latency column")
            return None

        lat_df = lat_df[latency_col]

        # ? Add the first latency values to the dataframe
        lat_df = pd.concat([pd.Series(first_latency_values), lat_df], axis=0)

        lat_df = lat_df.reset_index(drop=True)

        # Apply strip() to remove leading and trailing whitespaces
        lat_df = lat_df.apply(lambda x: x.strip() if isinstance(x, str) else x)

        # Set the type to float
        lat_df = pd.to_numeric(lat_df, errors="coerce")

        lat_df = lat_df.rename("latency_us")
        lat_df = lat_df.dropna()

        self.pub_df = lat_df

