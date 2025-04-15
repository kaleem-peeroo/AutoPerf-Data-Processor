import logging
import sys

from rich.logging import RichHandler
from typing import Dict

from app import App
from config import LD_DATASETS
from utils import clear_screen

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()]
    )

    ld_ds_config = validate_config(LD_DATASETS)

    o_a = App(ld_ds_config)
    o_a.run()

def validate_config(ld_config: Dict[str, str]) -> bool:
    raise NotImplementedError("validate_config is not implemented")

if __name__ == "__main__":
    clear_screen()
    main()
