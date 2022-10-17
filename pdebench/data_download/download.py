import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import logging

from easyDataverse import Dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="config/", config_name="config")
def main(config: DictConfig):
    """
    use config specifications to download files from a dataset
    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    # Change to original working directory

    os.chdir(get_original_cwd())

    # Extract dataset from the given DOI
    dataset = Dataset.from_dataverse_doi(
        doi=config.args.dataset_id,
        dataverse_url=config.args.dataverse_url,
        filenames=[config.args.filename],
        filedir=config.args.data_folder,
    )


if __name__ == "__main__":
    main()
