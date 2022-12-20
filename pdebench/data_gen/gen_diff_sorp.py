#!/usr/bin/env python

import os
import dotenv
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# this allows us to keep defaults local to the machine
# e.g. HPC versus local laptop
dotenv.load_dotenv()

import time

# or if the environment variables will be fixed for all executions, we can hard-code the environment variables like this:
num_threads = "4"

os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import dotenv
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
import multiprocessing as mp
from itertools import repeat
import numpy as np

from pdebench.data_gen.src import utils
import h5py
from pdebench.data_gen.uploader import dataverse_upload

log = logging.getLogger(__name__)


def simulator(config, i):
    
    from pdebench.data_gen.src import sim_diff_sorp
    
    config.sim.seed = i
    log.info(f"Starting seed {i}")
    start_time = time.time()
    sim_obj = sim_diff_sorp.Simulator(**config.sim)
    data_sample = sim_obj.generate_sample()
    duration = time.time() - start_time
    log.info(f"Seed {config.sim.seed} took {duration} to finish")
        
    seed_str = str(i).zfill(4)

    while True:
        try:
            with h5py.File(utils.expand_path(config.output_path), "a") as data_f:
                ## Chunking for compression and data access
                ## https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage
                ## should be by batch and less than 1MB
                ## lzf compression for float32 is kind of pointless though.
                data_f.create_dataset(
                    f"{seed_str}/data", data=data_sample, dtype="float32", compression="lzf"
                )
                data_f.create_dataset(
                    f"{seed_str}/grid/x", data = sim_obj.x, dtype="float32", compression="lzf"
                )
                data_f.create_dataset(
                    f"{seed_str}/grid/t", data=sim_obj.t, dtype="float32", compression="lzf" 
                )
                seed_group = data_f[seed_str]
                seed_group.attrs["config"] = OmegaConf.to_yaml(config)
        except IOError:
            time.sleep(0.1)
            continue
        else:
            break
    
    


@hydra.main(config_path="configs/", config_name="diff-sorp")
def main(config: DictConfig):
    """
    use config specifications to generate dataset
    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    
    # Change to original working directory to import modules
    
    temp_path = os.getcwd()
    os.chdir(get_original_cwd())
    
    from src import utils
    import h5py
    
    # Change back to the hydra working directory    
    os.chdir(temp_path)
    
    work_path = os.path.dirname(config.work_dir)
    output_path = os.path.join(work_path, config.data_dir, config.output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    config.output_path = os.path.join(output_path, config.output_path) + '.h5'
    
    num_samples_init = 0
    num_samples_final = 10000
    
    pool = mp.Pool(mp.cpu_count())
    seed = np.arange(num_samples_init, num_samples_final)
    seed = seed.tolist()
    pool.starmap(simulator, zip(repeat(config), seed))
    
    if config.upload:
        dataverse_upload(
            file_path=config.output_path,
            dataverse_url=os.getenv(
                'DATAVERSE_URL', 'https://darus.uni-stuttgart.de'),
            dataverse_token=os.getenv(
                'DATAVERSE_API_TOKEN', ''),
            dataverse_dir=config.name,
            dataverse_id=os.getenv(
                'DATAVERSE_ID', ''),
            log=log)

    return


import os

if __name__ == "__main__":
    test = main()
    