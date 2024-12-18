"""
Author : John Kim, Ran Zhang, Dan MacKinlay
PDE Simulation packages
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Optional

import imageio
import numpy as np
from pdebench.data_gen.src import data_io
from tqdm import tqdm

logger = logging.getLogger(__name__)

# import wandb


def call_many(fn_list, *args, **kwargs) -> list[callable]:
    """
    Surely there is already a helper function for this somewhere.
    inverse map?
    """
    return [fn(*args, **kwargs) for fn in fn_list]


def ns_sim(
    seed: int,
    label: Optional[str] = None,
    sim_name: str = "ns_sim_2d",
    particle_extrapolation: str = "BOUNDARY",
    velocity_extrapolation: str = "ZERO",
    NU: float = 0.01,
    scale: float = 10.0,
    smoothness: float = 3.0,
    grid_size=(100, 100),
    enable_gravity: bool = False,
    enable_obstacles: bool = False,
    force_extrapolation: str = "ZERO",
    save_images: bool = True,
    save_gif: bool = True,
    save_h5: bool = True,
    n_steps: int = 10,
    DT: float = 0.01,
    frame_int: int = 1,
    n_batch: int = 1,
    backend: str = "jax",
    device: str = "GPU",
    jit: bool = True,
    profile: bool = False,
    upload: bool = False,
    dataverse: Optional[dict] = None,
    config: dict | None = None,
):
    """
    Run the actual simulation.
    """

    if config is None:
        config = {}

    if backend == "jax":
        from phi.jax.flow import (
            Box,
            CenteredGrid,
            Noise,
            Obstacle,
            Solve,
            StaggeredGrid,
            advect,
            batch,
            diffuse,
            extrapolation,
            fluid,
            math,
        )
    elif backend == "pytorch":
        from phi.torch import TORCH
        from phi.torch.flow import (
            Box,
            CenteredGrid,
            Noise,
            Obstacle,
            Solve,
            StaggeredGrid,
            advect,
            batch,
            diffuse,
            extrapolation,
            fluid,
            math,
        )

        TORCH.set_default_device(device)
    else:
        from phi.flow import (
            Box,
            CenteredGrid,
            Noise,
            Obstacle,
            Solve,
            StaggeredGrid,
            advect,
            batch,
            diffuse,
            extrapolation,
            fluid,
            math,
        )

    # from torch.profiler import profile, record_function, ProfilerActivity

    def bounds_select(x: int | None = None, y: int | None = None) -> Box:
        """
        This function generates a 2D Phiflow Box with scale from zero to the number indicated by the bounds or infinite.

        Args:
            x (int): The bound in x axis
            y (int): The bound in y axis

        Returns:
            Box: A Box type of Phiflow
        """
        if x is None:
            return Box[:, 0:y]
        if y is None:
            return Box[0:x, :]
        return Box[0:x, 0:y]

    def cauchy_momentum_step(
        velocity, particles, body_acceleration, NU, DT, obstacles: tuple | None = None
    ) -> tuple[fluid.Field, fluid.Field]:
        """
        Navier-Stokes Simulation
        cauchy_momentum_step returns velocity and particles by solving cauchy momentum equation for one step
        cauchy_momentum_solve returns a list of velocity and particles for total simulation time

        Input variables
        velocity, particles : Observed variables (velocity & particles)
        body_acceleration : External body_acceleration terms
        NU : Fluid characteristics (kinematic viscosity)
        T, DT : Simulation time, interval (Simulation variables)
        **kwargs : Other obstacles (Simulation constraints etc)
        """
        # Set empty obstacle as empty tuple
        if obstacles is None:
            obstacles = ()
        # Computing velocity term first
        # Cauchy-momentum equation
        velocity = advect.semi_lagrangian(velocity, velocity, dt=DT)  # advection
        velocity = diffuse.explicit(velocity, NU, dt=DT)  # diffusion

        # Add external body_acceleration, constraints
        velocity += DT * particles * body_acceleration  # external body_acceleration
        velocity = fluid.apply_boundary_conditions(velocity, obstacles)  # obstacles

        # Make incompressible
        velocity, _ = fluid.make_incompressible(
            velocity, obstacles, solve=Solve("CG-adaptive", 1e-3, 0, x0=None)
        )  # particles

        # Computing particles term next
        particles = advect.semi_lagrangian(particles, velocity, dt=DT)

        return velocity, particles

    # Setting the random seed for simulation. This is a global seed for Phiflow.
    math.seed(seed)
    particles_images = []
    velocity_images = []
    callbacks = []
    cleanups = []
    if save_h5:
        data_store = data_io.h5_for(config)
        h5_path = data_store.filename

        def _store(frame_i, t, particles, velocity, **kwargs):
            data_store["particles"][:, frame_i, ...] = data_io.to_ndarray(particles)
            data_store["velocity"][:, frame_i, ...] = data_io.to_ndarray(velocity)
            data_store["t"][:, frame_i] = t
            data_store.attrs["latestIndex"] = frame_i

        callbacks.append(_store)
        cleanups.append(lambda *args, **kwargs: data_store.close())
        # Move output to artefacts dir here
        # if artefact_dir is not None:
        #     cleanups.append(
        #         lambda *args, **kwargs: data_store.close()
        #     )
        if upload:
            assert dataverse is not None
            cleanups.append(
                lambda *args, **kwargs: data_io.dataverse_upload(
                    file_path=h5_path,
                    dataverse_url=os.getenv(
                        "DATAVERSE_URL", "https://darus.uni-stuttgart.de"
                    ),
                    dataverse_token=os.getenv("DATAVERSE_API_TOKEN", ""),
                    dataverse_dir=label,
                    dataverse_id=dataverse["dataset_id"],
                )
            )

    else:
        data_store = data_io.dict_for(config)

        def _store(frame_i, t, particles, velocity, **kwargs):
            data_store["particles"][:, frame_i, ...] = data_io.to_ndarray(particles)
            data_store["velocity"][:, frame_i, ...] = data_io.to_ndarray(velocity)
            data_store["t"][:, frame_i] = t

        callbacks.append(_store)
    if save_images:

        def _save_img(frame_i, t, particles, velocity, **kwargs):
            particles_images.append()
            velocity_images.append()

        callbacks.append(_save_img)

    profile_ctx = nullcontext()
    if profile:
        profile_ctx = math.backend.profile(save="profile.json")

    with profile_ctx as prof:
        # Initialization of the particles (i.e. density of the flow) grid with a Phiflow Noise() method
        particles = CenteredGrid(
            Noise(batch(batch=n_batch), scale=scale, smoothness=smoothness),
            extrapolation=getattr(extrapolation, particle_extrapolation),
            x=grid_size[0],
            y=grid_size[1],
            bounds=bounds_select(*grid_size),
        )

        # Initialization of the velocity grid with a Phiflow Noise() method
        velocity = StaggeredGrid(
            Noise(batch(batch=n_batch), scale=scale, smoothness=smoothness),
            extrapolation=getattr(extrapolation, velocity_extrapolation),
            x=grid_size[0],
            y=grid_size[1],
            bounds=bounds_select(*grid_size),
        )

        # Initialization of the force grid. The force is also a staggered grid with a Phiflow Noise() method or using gravity
        if enable_gravity:
            force = math.tensor(batch(batch=n_batch), [0, -9.81])
        else:
            force = StaggeredGrid(
                Noise(batch(batch=n_batch), vector=2),
                extrapolation=getattr(extrapolation, force_extrapolation),
                x=grid_size[0],
                y=grid_size[1],
                bounds=bounds_select(*grid_size),
            )

        data_store["force"][:, ...] = data_io.to_ndarray(force)

        # Set the obstacles. Obstacles are not enabled by default.
        obstacles = []
        if enable_obstacles:
            obstacles.append(Obstacle(Box[45:55, 45:55]))
            obstacles.append(Obstacle(Box[15:25, 15:25]))

        # Use "python gen_ns_incomp.py save_images=false" for not saving to either pictures or .gif animation
        # Use "python gen_ns_incomp.py save_gif=false" for not saving .gif animation and only save to pictures

        call_many(
            callbacks,
            frame_i=0,
            t=0.0,
            velocity=velocity,
            particles=particles,
            prof=prof,
        )

        def sim_step(velocity, particles) -> tuple[fluid.Field, fluid.Field]:
            return cauchy_momentum_step(velocity, particles, force, NU, DT, obstacles)

        if jit:
            sim_step = math.jit_compile(sim_step)

        ts = np.linspace(0, n_steps * DT, n_steps, endpoint=False)
        n_steps_actual = ((n_steps - 1) // frame_int) * frame_int + 1
        ts = ts[1:n_steps_actual]

        for step, t in enumerate(tqdm(ts), start=1):
            velocity, particles = sim_step(
                velocity,
                particles,
            )

            if step % frame_int == 0:
                frame_i = step // frame_int
                msg = f"step {step} frame_i {frame_i}"
                logger.info(msg)
                call_many(
                    callbacks,
                    frame_i=frame_i,
                    t=t,
                    velocity=velocity,
                    particles=particles,
                    prof=prof,
                )

        if save_images and save_gif:
            # Saving images into .gif animation
            imageio.mimsave(
                f"{sim_name}_velocity.gif",
                velocity_images,
                duration=DT,
            )
            imageio.mimsave(
                f"{sim_name}_particles.gif",
                particles_images,
                duration=DT,
            )
        call_many(cleanups)
