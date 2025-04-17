Processes data from AutoPerf (AP).

# How to use
1. Populate `src/config.py` with list of campaigns.
2. Run `python src/main.py` to process the data.

# How the config works
Each campaign must be a dictionary with the following keys:
- name: required name of campaign
- exp_folders: required path to folder containing experiments
- ap_config: optional path to AutoPerf config file used to generate the experiments
- dataset_path: required path to dataset file to make (must end with .parquet)
