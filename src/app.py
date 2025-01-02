import toml

from logger import logger
from .campaign import Campaign

from rich.console import Console
from rich.pretty import pprint
console = Console()

class App:
    usage_message = "You should have 3 args. Usage: \n\t{}".format(
        "python run.py <input_data_dir> <ap_config_path> <output_dataset_path>"
    )

    def __init__(self, args):
        self.args = args

    def read_config(self):
        try:
            with open("config.toml") as f:
                config = toml.load(f)

        except Exception as e:
            logger.error("Error reading config.toml")
            logger.error(e)
            return None

        if 'datasets' not in config:
            raise Exception("No datasets key in config.toml")

        datasets = config['datasets']
        if not isinstance(datasets, list):
            raise Exception("Datasets key in config.toml is not a list")

        if len(datasets) == 0:
            raise Exception("Datasets key in config.toml is empty")

        return datasets

    def run(self):
        if len(self.args) != 0 and len(self.args) != 3:
            logger.critical(self.usage_message)
            return

        if len(self.args) == 3:
            data_dir = self.args[0]
            apconf_path = self.args[1]
            output_path = self.args[2]

            logger.info("Running with the following args:")
            logger.info(f"\tdata_dir: {data_dir}")
            logger.info(f"\tapconf_path: {apconf_path}")
            logger.info(f"\toutput_path: {output_path}")

            campaign = Campaign(data_dir, apconf_path)

            if not campaign.validate_args():
                raise Exception("Invalid campaign args")

            campaign.generate_config()
            incomplete_exp_names = campaign.create_dataset(output_path)
            campaign.generate_missing_test_config(incomplete_exp_names)            

        else:
            console.print("Using config.toml.", style="bold green")
            datasets = self.read_config()
            if datasets is None:
                raise Exception("datasets is None after reading config.toml")

            for dataset in datasets:
                if 'name' not in dataset:
                    logger.error("No name key in dataset")
                    continue

                if 'exp_folders' not in dataset:
                    logger.error("No data_dir key in dataset")
                    continue

                if 'ap_config' not in dataset:
                    logger.error("No apconf_path key in dataset")
                    continue

                if 'dataset_path' not in dataset:
                    logger.error("No output_path key in dataset")
                    continue

                ds_name = dataset['name']
                data_dir = dataset['exp_folders']
                apconf_path = dataset['ap_config']
                output_path = dataset['dataset_path']

                logger.info(f"Processing {ds_name} with the following args:")
                logger.info(f"\tdata_dir: {data_dir}")
                logger.info(f"\tapconf_path: {apconf_path}")
                logger.info(f"\toutput_path: {output_path}")

                campaign = Campaign(data_dir, apconf_path)

                if not campaign.validate_args():
                    raise Exception("Invalid campaign args")

                campaign.generate_config()
                incomplete_exp_names = campaign.create_dataset(output_path)
                campaign.generate_missing_test_config(incomplete_exp_names)            
