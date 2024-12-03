Processes data from AutoPerf (AP).

Inputs:
- path to folders containing AP .csv files
- path to output folder where the dataset will be saved
- path to the AP config file that generated the campaign data

Outputs:
- new AP config file with a list of missing tests
- dataset as a .parquet

Usage:
```python
python run.py <exp_folders> <ap_config> <dataset_path>
```

Where:
- `<exp_folders>` is a list of paths to folders containing AP .csv files
- `<ap_config>` is the path to the AP config file that generated the campaign data
- `<dataset_path>` is the path to the output folder where the dataset will be saved
