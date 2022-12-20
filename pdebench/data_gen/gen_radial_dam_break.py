#!/usr/bin/env python

from copy import deepcopy
import os

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# this allows us to keep defaults local to the machine
# e.g. HPC versus local laptop
import dotenv

dotenv.load_dotenv()

# or if the environment variables will be fixed for all executions, we can hard-code the environment variables like this:
num_threads = "4"

os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_MAX_THREADS"] = num_threads

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import h5py
import logging
import multiprocessing as mp
from itertools import repeat
from pdebench.data_gen.src import utils
import numpy as np
from pdebench.data_gen.uploader import dataverse_upload
import time
from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D

log = logging.getLogger(__name__)


def simulator(base_config, i):
    
    
    config = deepcopy(base_config)
    config.sim.seed = i
    log.info(f"Starting seed {i}")

    np.random.seed(config.sim.seed)
    # config.sim.inner_height = np.random.uniform(1.5, 2.5)
    config.sim.dam_radius = np.random.uniform(0.3, 0.7)

    scenario = RadialDamBreak2D(
        grav=config.sim.gravity,
        dam_radius=config.sim.dam_radius,
        xdim=config.sim.xdim,
        ydim=config.sim.ydim,
    )

    start_time = time.time()
    scenario.run(T=config.sim.T_end, tsteps=config.sim.n_time_steps, plot=False)
    duration = time.time() - start_time
    seed_str = str(i).zfill(4)
    log.info(f"Seed {seed_str} took {duration} to finish")
    
    while True:
        try:
            with h5py.File(utils.expand_path(config.output_path), "a") as h5_file:
                scenario.save_state_to_disk(h5_file, seed_str)
                seed_group = h5_file[seed_str]
                seed_group.attrs["config"] = OmegaConf.to_yaml(config)
        except IOError:
            time.sleep(0.1)
            continue
        else:
            break

    if config.upload:
        dataverse_upload(
            file_path=config.output_path,
            dataverse_url=os.getenv("DATAVERSE_URL", "https://darus.uni-stuttgart.de"),
            dataverse_token=os.getenv("DATAVERSE_API_TOKEN", ""),
            dataverse_dir=config.name,
            dataverse_id=os.getenv("DATAVERSE_ID", ""),
            log=log,
        )
        

@hydra.main(config_path="configs/", config_name="radial_dam_break")
def main(config: DictConfig):
    """
    use config specifications to generate dataset
    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    # Change to original working directory to import modules
    import os

    temp_path = os.getcwd()
    os.chdir(get_original_cwd())
    
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

    return


if __name__ == "__main__":
    main()
