import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import logging

from pyDataverse.api import NativeApi
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
    os.environ["DATAVERSE_URL"] = config.args.dataverse_url

    # Extract dataset from the given DOI
    dataset = Dataset()
    setattr(dataset, "p_id", config.args.dataset_id)

    # Extract file list contained in the dataset
    api = NativeApi(config.args.dataverse_url)
    dv_dataset = api.get_dataset(config.args.dataset_id)
    files_list = dv_dataset.json()["data"]["latestVersion"]["files"]

    # Compile list of files that matches the desired filename
    files = []
    for i, file in enumerate(files_list):
        if config.args.filename in file["dataFile"]["filename"]:
            files.append(file["dataFile"]["filename"])

    # Download the files
    
    dataset = Dataset.from_dataverse_doi(
        doi=config.args.dataset_id,
        dataverse_url=config.args.dataverse_url,
        filenames=files,
        filedir=config.args.data_folder,
    )



if __name__ == "__main__":
    main()
