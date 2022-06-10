import os
import dotenv

dotenv.load_dotenv()

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import logging

import glob
from pyDataverse.api import NativeApi, DataAccessApi
from pyDaRUS import Dataset
from easyDataverse.core.downloader import download_files

log = logging.getLogger(__name__)


@hydra.main(config_path="config/", config_name="config")
def main(config: DictConfig):
    """
    use config specifications to generate dataset
    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    # Change to original working directory

    os.chdir(get_original_cwd())

    path_to_data = os.path.join(os.path.abspath("../data"), config.args.filename)

    file_list = sorted(glob.glob(path_to_data + "/*.h5"))
    doi = "doi:10.18419/darus-2986"

    # Extract dataset from the given DOI
    dataset = Dataset()
    setattr(dataset, "p_id", doi)

    # Extract file list contained in the dataset
    api = NativeApi(os.getenv("DATAVERSE_URL"), os.getenv("DATAVERSE_API_TOKEN"))
    data_api = DataAccessApi(
        os.getenv("DATAVERSE_URL"), os.getenv("DATAVERSE_API_TOKEN")
    )
    dv_dataset = api.get_dataset(doi)
    files_list = dv_dataset.json()["data"]["latestVersion"]["files"]

    # Compile list of files that matches the desired filename
    files = []
    for i, file in enumerate(files_list):
        if config.args.filename in file["dataFile"]["filename"]:
            files.append(file)

    # Download the files
    download_files(data_api, dataset, files, os.path.abspath("../data/"))

    return


import os

if __name__ == "__main__":
    main()
