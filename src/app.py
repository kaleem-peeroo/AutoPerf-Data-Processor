from logger import logger
from .campaign import Campaign

class App:
    usage_message = "You should have 3 args. Usage: \n\t{}".format(
        "python run.py <input_data_dir> <ap_config_path> <output_dataset_path>"
    )

    def __init__(self, args):
        self.args = args

    def run(self):
        if len(self.args) == 0:
            logger.critical(self.usage_message)
            return

        if len(self.args) != 3:
            logger.critical(self.usage_message)
            return

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
