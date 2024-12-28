"""
Created on Wed May  4 09:53:18 2022

@author: timot
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pdebench.data_gen.src.plots import plot_data


@hydra.main(config_path="configs/", config_name="diff-sorp")
def main(config: DictConfig):
    """
    use config specifications to generate dataset
    """
    work_path = Path(config.work_dir)
    output_path = work_path / config.data_dir / config.output_path
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    config.output_path = output_path / config.output_path

    # Open and load file
    data_path = config.output_path + ".h5"
    h5_file = h5py.File(data_path, "r")
    rng = np.random.default_rng()

    if "seed" in config.sim:
        # Choose random sample number
        idx_max = 10000 if config.plot.dim == 1 else 1000
        config.sim.seed = rng.integers(0, idx_max)
        postfix = str(config.sim.seed).zfill(4)
        data = np.array(h5_file[f"{postfix}/data"], dtype="f")
        t = np.array(h5_file[f"{postfix}/grid/t"], dtype="f")
        # data dim = [t, x1, ..., xd, v]
    else:
        idx_max = 10000 if config.plot.dim == 1 else 1000
        postfix = rng.randint(0, idx_max)
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


if __name__ == "__main__":
    main()
