from ctypes import memmove
import os
import psutil
import logging
import pandas as pd

# from logger import logger
from rich.pretty import pprint

logger = logging.getLogger(__name__)

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_qos_name(qos_settings):
    if not isinstance(qos_settings, dict):
        raise ValueError(f"QoS settings must be a dict: {qos_settings}")

    if qos_settings['use_reliable']:
        use_reliable_str = "REL"
    else:
        use_reliable_str = "BE"

    if qos_settings['use_multicast']:
        use_multicast_str = "MC"
    else:
        use_multicast_str = "UC"

    return "{}SEC_{}B_{}P_{}S_{}_{}_{}DUR_{}LC".format(
        qos_settings['duration_secs'],
        qos_settings['datalen_bytes'],
        qos_settings['pub_count'],
        qos_settings['sub_count'],
        use_reliable_str,
        use_multicast_str,
        qos_settings['durability'],
        qos_settings['latency_count']
    ) 

def aggregate_across_cols(df, agg_types=["avg", "total"]):
    if df is None:
        raise ValueError("df must not be None")

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a DataFrame: {df}")

    if len(df) == 0:
        raise ValueError("df must not be empty")

    if not isinstance(agg_types, list):
        raise ValueError(f"agg_types must be a list: {agg_types}")

    if len(agg_types) == 0:
        raise ValueError("agg_types must not be empty")
    
    metrics = [
        'mbps', 
        'total_samples', 
        'lost_samples'
        'lost_samples_percent', 
        'samples_per_sec', 
    ]

    desired_cols = [
        'experiment_name',
        'duration_secs',
        'datalen_bytes',
        'pub_count',
        'sub_count',
        'use_reliable',
        'use_multicast',
        'durability',
        'latency_count',
        'latency_us'
    ]

    if len(df.columns) == 0:
        raise ValueError("df must have columns")

    for desired_col in desired_cols:
        if desired_col not in df.columns:
            raise ValueError(f"{desired_col} not found in df columns: {df.columns}")
    
    df_cols = [col for col in df.columns if col != 0]
    new_df = pd.DataFrame()

    for metric in metrics:
        metric_cols = [col for col in df_cols if metric in col]

        for col in metric_cols:
            try:
                df[col] = df[col].astype(float, errors='ignore')
            except ValueError as e:
                logger.error(f"Error converting {col} to float: {e}")
                continue

        col_dtypes = [str(item) for item in df[metric_cols].dtypes.to_dict().values()]
        if "object" in col_dtypes:
            logger.error(f"Skipping {metric} because it contains non-numeric values")
            continue

        for agg_type in agg_types:
            if agg_type == "avg":
                df[f"{agg_type}_{metric}"] = df[metric_cols].mean(axis=1)

            else:
                df[f"{agg_type}_{metric}"] = df[metric_cols].sum(axis=1)

            desired_cols.append(f"{agg_type}_{metric}")

    for col in desired_cols:
        new_df[col] = df[col]

    return new_df

def calculate_averages(df):
    metrics = [
        'mbps', 
        'total_samples', 
        'lost_samples'
        'lost_samples_percent', 
        'samples_per_sec', 
    ]

    desired_cols = [
        'experiment_name',
        'duration_secs',
        'datalen_bytes',
        'pub_count',
        'sub_count',
        'use_reliable',
        'use_multicast',
        'durability',
        'latency_count',
        'latency_us'
    ]

    df_cols = [col for col in df.columns if col != 0]
    new_df = pd.DataFrame()

    for metric in metrics:
        metric_cols = [col for col in df_cols if metric in col]

        for col in metric_cols:
            try:
                df[col] = df[col].astype(float, errors='ignore')
            except ValueError as e:
                logger.error(f"Error converting {col} to float: {e}")
                continue

        col_dtypes = [str(item) for item in df[metric_cols].dtypes.to_dict().values()]
        if "object" in col_dtypes:
            logger.error(f"Skipping {metric} because it contains non-numeric values")
            continue

        df[f"avg_{metric}"] = df[metric_cols].mean(axis=1)
        desired_cols.append(f"avg_{metric}")

    for col in desired_cols:
        new_df[col] = df[col]

    return new_df

def get_df_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return None

    qos = get_qos_from_name(os.path.basename(csv_path).replace(".csv", ""))

    avg_mbps_cols = [col for col in df.columns if 'mbps' in col and 'avg' in col]

    if 'latency_us' not in df.columns and len(avg_mbps_cols) == 0:
        logger.error(f"No latency or avg throughput columns found in {csv_path}")
        return None

    exp_df = pd.DataFrame()

    avg_cols = [col for col in df.columns if 'avg' in col]
    for col in avg_cols:
        exp_df[col] = df[col]

    exp_df['latency_us'] = df['latency_us']

    for key, value in qos.items():
        exp_df[key] = value

    exp_df['experiment_name'] = os.path.basename(csv_path).replace(".csv", "")

    return exp_df

def get_qos_from_name(name):
    duration_secs = 0
    datalen_bytes = 0
    pub_count = 0
    sub_count = 0
    use_reliable = False
    use_multicast = False
    durability = 0
    latency_count = 0

    if name == "":
        raise ValueError("experiment name must not be empty")

    if not isinstance(name, str):
        raise ValueError(f"experiment name must be a string: {name}")

    name = name.replace("PUB_", "P_")
    name = name.replace("SUB_", "S_")

    name_parts = name.split("_")
    if len(name_parts) != 8:
        raise ValueError("{} must have 8 parts but has {}".format(
            name, len(name_parts)
        ))

    for part in name_parts:
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

def mem_usage(units="mb"):
    if units not in ["mb", "gb"]:
        raise ValueError(f"Invalid units: {units}")

    mem = psutil.virtual_memory().used

    if units == "b":
        return mem

    elif units == "kb":
        return mem / 1024

    elif units == "mb":
        return mem / 1024 / 1024

    elif units == "gb":
        return mem / 1024 / 1024 / 1024

    else:
        raise ValueError(f"Invalid units: {units}")

