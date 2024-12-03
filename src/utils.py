import pandas as pd

from rich.pretty import pprint

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
        df[f"avg_{metric}"] = df[metric_cols].mean(axis=1)
        desired_cols.append(f"avg_{metric}")

    for col in desired_cols:
        new_df[col] = df[col]

    return new_df
