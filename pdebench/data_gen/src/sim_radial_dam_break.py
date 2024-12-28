from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from clawpack import pyclaw, riemann

logger = logging.getLogger(__name__)


class Basic2DScenario(ABC):
    name = ""

    def __init__(self):
        self.solver = None
        self.claw_state = None
        self.domain = None
        self.solution = None
        self.claw = None
        self.save_state = {}
        self.state_getters = {}

        self.setup_solver()
        self.create_domain()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.register_state_getters()
        self.outdir = Path.cwd() / (self.name.replace(" ", "") + "2D")

    @abstractmethod
    def setup_solver(self):
        pass

    @abstractmethod
    def create_domain(self):
        pass

    @abstractmethod
    def set_initial_conditions(self):
        pass

    @abstractmethod
    def set_boundary_conditions(self):
        pass

    def __get_h(self):
        return self.claw_state.q[self.depthId, :].tolist()

    def __get_u(self):
        return (
            self.claw_state.q[self.momentumId_x, :] / self.claw_state.q[self.depthId, :]
        ).tolist()

    def __get_v(self):
        return (
            self.claw_state.q[self.momentumId_y, :] / self.claw_state.q[self.depthId, :]
        ).tolist()

    def __get_hu(self):
        return self.claw_state.q[self.momentumId_x, :].tolist()

    def __get_hv(self):
        return self.claw_state.q[self.momentumId_y, :].tolist()

    def register_state_getters(self) -> None:
        self.state_getters = {
            "h": self.__get_h,
            "u": self.__get_u,
            "v": self.__get_v,
            "hu": self.__get_hu,
            "hv": self.__get_hv,
        }

    def add_save_state(self) -> None:
        for key, getter in self.state_getters.items():
            self.save_state[key].append(getter())

    def init_save_state(self, T: float, tsteps: int) -> None:
        self.save_state = {}
        self.save_state["x"] = self.domain.grid.x.centers.tolist()
        self.save_state["y"] = self.domain.grid.y.centers.tolist()
        self.save_state["t"] = np.linspace(0.0, T, tsteps + 1).tolist()
        for key, getter in self.state_getters.items():
            self.save_state[key] = [getter()]

    def save_state_to_disk(self, data_f, seed_str) -> None:
        T = np.asarray(self.save_state["t"])
        X = np.asarray(self.save_state["x"])
        Y = np.asarray(self.save_state["y"])
        H = np.expand_dims(np.asarray(self.save_state["h"]), -1)

        data_f.create_dataset(f"{seed_str}/data", data=H, dtype="f")
        data_f.create_dataset(f"{seed_str}/grid/x", data=X, dtype="f")
        data_f.create_dataset(f"{seed_str}/grid/y", data=Y, dtype="f")
        data_f.create_dataset(f"{seed_str}/grid/t", data=T, dtype="f")

    def simulate(self, t) -> None:
        if all(v is not None for v in [self.domain, self.claw_state, self.solver]):
            self.solver.evolve_to_time(self.solution, t)
        else:
            msg = "Simulate failed: No scenario defined."
            logger.info(msg)

    def run(self, T: float = 1.0, tsteps: int = 20) -> None:
        self.init_save_state(T, tsteps)
        self.solution = pyclaw.Solution(self.claw_state, self.domain)
        dt = T / float(tsteps)

        for tstep in range(1, tsteps + 1):
            t = tstep * dt
            self.simulate(t)
            self.add_save_state()


class RadialDamBreak2D(Basic2DScenario):
    name = "RadialDamBreak"

    def __init__(
        self,
        xdim,
        ydim,
        grav: float = 1.0,
        dam_radius: float = 0.5,
        inner_height: float = 2.0,
    ):
        self.depthId = 0
        self.momentumId_x = 1
        self.momentumId_y = 2
        self.grav = grav
        self.xdim = xdim
        self.ydim = ydim
        self.dam_radius = dam_radius
        self.inner_height = inner_height
        super().__init__()
        # self.state_getters['bathymetry'] = self.__get_bathymetry

    def setup_solver(self) -> None:
        rs = riemann.shallow_roe_with_efix_2D
        self.solver = pyclaw.ClawSolver2D(rs)
        self.solver.limiters = pyclaw.limiters.tvd.MC
        # self.solver.fwave = True
        self.solver.num_waves = 3
        self.solver.num_eqn = 3
        self.depthId = 0
        self.momentumId_x = 1
        self.momentumId_y = 2

    def create_domain(self) -> None:
        self.xlower = -2.5
        self.xupper = 2.5
        self.ylower = -2.5
        self.yupper = 2.5
        mx = self.xdim
        my = self.ydim
        x = pyclaw.Dimension(self.xlower, self.xupper, mx, name="x")
        y = pyclaw.Dimension(self.ylower, self.yupper, my, name="y")
        self.domain = pyclaw.Domain([x, y])
        self.claw_state = pyclaw.State(self.domain, self.solver.num_eqn)

    def set_boundary_conditions(self) -> None:
        """
        Sets homogeneous Neumann boundary conditions at each end for q=(u, h*u)
        and for the bathymetry (auxiliary variable).
        """
        self.solver.bc_lower[0] = pyclaw.BC.extrap
        self.solver.bc_upper[0] = pyclaw.BC.extrap
        self.solver.bc_lower[1] = pyclaw.BC.extrap
        self.solver.bc_upper[1] = pyclaw.BC.extrap

    def initial_h(self, coords):
        x0 = 0.0
        y0 = 0.0
        x = coords[:, 0]
        y = coords[:, 1]
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        h_in = self.inner_height
        h_out = 1.0
        return h_in * (r <= self.dam_radius) + h_out * (r > self.dam_radius)

    @staticmethod
    def initial_momentum_x() -> torch.Tensor:
        return torch.tensor(0.0)

    @staticmethod
    def initial_momentum_y() -> torch.Tensor:
        return torch.tensor(0.0)

    def __get_bathymetry(self):
        return self.claw_state.aux[0, :].tolist()

    def set_initial_conditions(self) -> None:
        self.claw_state.problem_data["grav"] = self.grav

        xc = self.claw_state.grid.x.centers
        xc = torch.tensor(xc)

        x0 = 0.0
        y0 = 0.0
        X, Y = self.claw_state.p_centers
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        h_in = 2.0
        h_out = 1.0

        self.claw_state.q[self.depthId, :, :] = h_in * (
            r <= self.dam_radius
        ) + h_out * (r > self.dam_radius)
        self.claw_state.q[self.momentumId_x, :, :] = 0.0
        self.claw_state.q[self.momentumId_y, :, :] = 0.0


if __name__ == "__main__":
    # run simulation based on the given scenario
    scenario = RadialDamBreak2D(xdim=64, ydim=64)
    scenario.run(tsteps=100, plot=False)
    scenario.save_state_to_disk()
