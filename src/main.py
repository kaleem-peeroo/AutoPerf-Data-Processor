import logging
import sys
import os

from rich.logging import RichHandler
from rich.pretty import pprint
from typing import Dict, List

from app import App
from campaign import Campaign
from config import LD_DATASETS
from utils import clear_screen

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()]
    )
    lg = logging.getLogger(__name__)

    ld_ds_config = validate_config(LD_DATASETS)

    for i_ds, d_ds in enumerate(ld_ds_config):
        lg.info("[{}/{}] Processing {}".format(
            i_ds + 1, len(ld_ds_config), d_ds['name']
        ))
        campaign = Campaign(d_ds)
        campaign.create_dataset()

def validate_config(
    ld_config: List[ Dict[str, str] ] = []
) -> List[ Dict[str, str] ]:
    if ld_config is None:
        raise ValueError("ld_config must not be None")

    if not isinstance(ld_config, list):
        raise ValueError(f"ld_config must be a list: {ld_config}")

    if len(ld_config) == 0:
        raise ValueError("ld_config must not be empty")

    for d_config in ld_config:
        if not isinstance(d_config, dict):
            raise ValueError(f"d_config must be a dict: {d_config}")

        required_keys = [
            'name', 
            'exp_folders', 
            'ap_config', 
            'dataset_path'
        ]

        for key in required_keys:
            if key not in d_config:
                raise ValueError(f"{key} not found in d_config")

        for key in d_config:
            if key not in required_keys:
                raise ValueError(f"{key} is not a valid key in d_config")

        """
        name: str
        exp_folders: str: valid path to folder that exists
        ap_config: str: optional path to AP config file
        dataset_path: str: optional path to .parquet file to be created
        """
        if not isinstance(d_config['name'], str):
            raise ValueError(f"name must be a str: {d_config['name']}")

        if not isinstance(d_config['exp_folders'], str):
            raise ValueError(f"exp_folders must be a str: {d_config['exp_folders']}")

        if not isinstance(d_config['ap_config'], str):
            raise ValueError(f"ap_config must be a str: {d_config['ap_config']}")

        if not isinstance(d_config['dataset_path'], str):
            raise ValueError(f"dataset_path must be a str: {d_config['dataset_path']}")

        if not os.path.exists(d_config['exp_folders']):
            raise ValueError(f"exp_folders does not exist: {d_config['exp_folders']}")

        if not d_config['dataset_path'].endswith('.parquet'):
            raise ValueError(
                f"dataset_path must be a .parquet file: {d_config['dataset_path']}"
            )

    return ld_config
    
if __name__ == "__main__":
    clear_screen()
    main()
