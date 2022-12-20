# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:53:18 2022

@author: timot
"""


import hydra
from omegaconf import DictConfig
import numpy as np
import h5py
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd
from pdebench.data_gen.src.plots import plot_data

@hydra.main(config_path="configs/", config_name="diff-sorp")
def main(config: DictConfig):
    """
    use config specifications to generate dataset
    """

    work_path = os.path.dirname(config.work_dir)
    output_path = os.path.join(work_path, config.data_dir, config.output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    config.output_path = os.path.join(output_path, config.output_path)
    
    # Open and load file
    data_path = config.output_path + '.h5'
    h5_file = h5py.File(data_path, "r")
    
    if "seed" in config.sim.keys():
        # Choose random sample number
        idx_max = 10000 if config.plot.dim == 1 else 1000
        config.sim.seed = np.random.randint(0, idx_max)
        postfix = str(config.sim.seed).zfill(4)
        data = np.array(h5_file[f"{postfix}/data"], dtype="f")
        t = np.array(h5_file[f"{postfix}/grid/t"], dtype="f")
        # data dim = [t, x1, ..., xd, v]
    else:
        idx_max = 10000 if config.plot.dim == 1 else 1000
        postfix = np.random.randint(0, idx_max)
        data = np.array(h5_file["data"], dtype="f")
        data = data[postfix]
        t = np.array(h5_file["grid/t"], dtype="f")
        t = t[postfix]
        # data dim = [t, x1, ..., xd, v]
    
    h5_file.close()

    os.chdir(get_original_cwd())
    plot_data(
        data,
        t,
        config.plot.dim,
        config.plot.channel_idx,
        config.plot.t_idx,
        config,
        config.name + "_" + postfix + ".png",
    )

    return


import os

if __name__ == "__main__":
    main()
