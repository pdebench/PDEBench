import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import logging

from pyDataverse.api import NativeApi, DataAccessApi
from pyDaRUS import Dataset
from easyDataverse.core.downloader import download_files

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
    dataset = Dataset()
    setattr(dataset, "p_id", config.args.dataset_id)

    # Extract file list contained in the dataset
    api = NativeApi(config.args.dataverse_url)
    data_api = DataAccessApi(config.args.dataverse_url)
    dv_dataset = api.get_dataset(config.args.dataset_id)
    files_list = dv_dataset.json()["data"]["latestVersion"]["files"]

    # Compile list of files that matches the desired filename
    files = []
    for i, file in enumerate(files_list):
        if config.args.filename in file["dataFile"]["filename"]:
            files.append(file)

    # Download the files
    download_files(data_api, dataset, files, os.path.abspath(config.args.data_folder))


if __name__ == "__main__":
    main()
