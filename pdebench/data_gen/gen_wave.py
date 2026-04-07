"""
Data generation script for the wave equation and Klein-Gordon equation.

Wave equation: d2u/dt2 = c^2 * Lap(u)
Klein-Gordon:  d2u/dt2 = c^2 * Lap(u) - chi^2 * u

Uses leapfrog finite-difference integration with periodic BCs on [0,1]^d.
Initial conditions are random Fourier superpositions.

Usage:
    python gen_wave.py                        # Default: 1D wave, c=1.0
    python gen_wave.py sim.c=0.4              # Change wave speed
    python gen_wave.py sim.chi=1.0            # Klein-Gordon with mass=1
    python gen_wave.py sim.ndim=2 sim.xdim=128  # 2D wave equation

Output format: HDF5 with PDEBench Format 1A
    - "tensor": shape (n_samples, t, x) for 1D or (n_samples, t, x, y) for 2D
    - "x-coordinate": spatial grid coordinates
    - "t-coordinate": temporal grid coordinates
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from itertools import repeat
from pathlib import Path

import dotenv
import h5py
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

dotenv.load_dotenv()

num_threads = "4"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads

log = logging.getLogger(__name__)


def simulator(config: DictConfig, seed: int) -> None:
    """Generate one sample and write to HDF5."""
    from pdebench.data_gen.src import sim_wave

    sim_config = dict(config.sim)
    sim_config["seed"] = seed

    log.info(f"Starting seed {seed}")
    start_time = time.time()

    sim = sim_wave.WaveSimulator(**sim_config)
    data_sample = sim.generate_sample()  # shape: (Nt, Nx) or (Nt, Nx, Nx)

    duration = time.time() - start_time
    log.info(f"Seed {seed} took {duration:.2f}s")

    seed_str = str(seed).zfill(4)

    while True:
        try:
            with h5py.File(str(config.output_path), "a") as f:
                f.create_dataset(
                    f"{seed_str}/data",
                    data=data_sample,
                    dtype="float32",
                    compression="lzf",
                )
                f.create_dataset(
                    f"{seed_str}/grid/x",
                    data=sim.x.astype(np.float32),
                    dtype="float32",
                    compression="lzf",
                )
                f.create_dataset(
                    f"{seed_str}/grid/t",
                    data=sim.t_save.astype(np.float32),
                    dtype="float32",
                    compression="lzf",
                )
                if config.sim.ndim == 2:
                    f.create_dataset(
                        f"{seed_str}/grid/y",
                        data=sim.x.astype(np.float32),
                        dtype="float32",
                        compression="lzf",
                    )
                seed_group = f[seed_str]
                seed_group.attrs["config"] = OmegaConf.to_yaml(config)
        except OSError:
            time.sleep(0.1)
            continue
        else:
            break


def combine_to_tensor_format(
    h5_path: Path,
    output_path: Path,
    n_samples: int,
) -> None:
    """
    Convert per-seed HDF5 to PDEBench Format 1A
    (single "tensor" dataset with batch dimension).
    """
    with h5py.File(str(h5_path), "r") as f_in:
        # Get shape from first sample
        first_key = str(0).zfill(4)
        sample_shape = f_in[f"{first_key}/data"].shape

        x_coord = np.array(f_in[f"{first_key}/grid/x"])
        t_coord = np.array(f_in[f"{first_key}/grid/t"])
        y_coord = None
        if f"{first_key}/grid/y" in f_in:
            y_coord = np.array(f_in[f"{first_key}/grid/y"])

        # Allocate combined tensor
        full_shape = (n_samples, *sample_shape)

        with h5py.File(str(output_path), "w") as f_out:
            tensor = f_out.create_dataset(
                "tensor",
                shape=full_shape,
                dtype="float32",
                compression="lzf",
            )
            for i in range(n_samples):
                key = str(i).zfill(4)
                if key not in f_in:
                    msg = (
                        f"Missing seed {key} in {h5_path}; "
                        f"expected {n_samples} consecutive seeds "
                        f"0000..{str(n_samples - 1).zfill(4)}"
                    )
                    raise KeyError(msg)
                tensor[i] = f_in[f"{key}/data"]

            f_out.create_dataset("x-coordinate", data=x_coord)
            f_out.create_dataset("t-coordinate", data=t_coord)
            if y_coord is not None:
                f_out.create_dataset("y-coordinate", data=y_coord)

    log.info(f"Combined tensor format saved to {output_path} with shape {full_shape}")


@hydra.main(config_path="configs/", config_name="wave", version_base="1.1")
def main(config: DictConfig):
    """Generate wave equation dataset using Hydra config."""
    temp_path = Path.cwd()
    os.chdir(get_original_cwd())
    os.chdir(temp_path)

    work_path = Path(config.work_dir)
    output_dir: Path = work_path / config.data_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    chi_str = f"_chi{config.sim.chi}" if config.sim.chi > 0 else ""
    base_name = f"{config.sim.ndim}D_Wave_c{config.sim.c}{chi_str}"
    config.output_path = str((output_dir / base_name).with_suffix(".h5"))

    num_samples = config.num_samples

    log.info(f"Generating {num_samples} samples -> {config.output_path}")
    log.info(f"PDE: d2u/dt2 = {config.sim.c}^2 * Lap(u) - {config.sim.chi}^2 * u")

    # Generate samples in parallel
    pool = mp.Pool(mp.cpu_count())
    seeds = list(range(num_samples))
    pool.starmap(simulator, zip(repeat(config), seeds))
    pool.close()
    pool.join()

    # Convert to PDEBench Format 1A ("tensor" key)
    raw_path = Path(config.output_path)
    tensor_path = raw_path.parent / f"{base_name}_tensor.hdf5"
    combine_to_tensor_format(raw_path, tensor_path, num_samples)

    log.info("Done.")


if __name__ == "__main__":
    main()
