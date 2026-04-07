"""
Simulator for the 1D/2D wave equation and Klein-Gordon equation.

Wave equation:
    d2u/dt2 = c^2 * Lap(u)

Klein-Gordon equation:
    d2u/dt2 = c^2 * Lap(u) - chi^2 * u

Solved using second-order leapfrog (Verlet) finite-difference integration
with periodic boundary conditions on domain [0, 1]^d.

Initial conditions are random superpositions of Fourier modes.
Analytical solutions are computed via Fourier decomposition for validation.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class WaveSimulator:
    """
    Simulator for the wave equation in 1D or 2D.

    Uses leapfrog (Verlet) time integration, second-order in both
    space and time. Periodic boundary conditions.

    :param c: wave speed
    :param chi: mass parameter (0 = wave equation, >0 = Klein-Gordon)
    :param t: stop time
    :param tdim: number of time steps to save
    :param xdim: number of spatial grid points per dimension
    :param ndim: spatial dimensionality (1 or 2)
    :param n_modes: number of Fourier modes in IC generation
    :param n: number of batches (unused; present for API compatibility)
    :param seed: random seed for IC generation
    """

    def __init__(
        self,
        c: float = 1.0,
        chi: float = 0.0,
        t: float = 2.0,
        tdim: int = 201,
        xdim: int = 1024,
        ndim: int = 1,
        n_modes: int = 5,
        n: int = 1,  # noqa: ARG002
        seed: int = 0,
    ):
        self.c = c
        self.chi = chi
        self.T = t
        self.Nt = tdim
        self.Nx = xdim
        self.ndim = ndim
        self.n_modes = n_modes
        self.seed = seed

        # Spatial grid
        self.dx = 1.0 / self.Nx
        if ndim == 1:
            self.x = np.linspace(0, 1, self.Nx, endpoint=False, dtype=np.float64)
        elif ndim == 2:
            x1d = np.linspace(0, 1, self.Nx, endpoint=False, dtype=np.float64)
            self.x = x1d  # store 1D coordinate
            self.X, self.Y = np.meshgrid(x1d, x1d, indexing="ij")
        else:
            errmsg = f"ndim must be 1 or 2, got {ndim}"
            raise ValueError(errmsg)

        # Time grid
        self.t_save = np.linspace(0, self.T, self.Nt, dtype=np.float64)

        # CFL condition: dt < dx / (c * sqrt(ndim))
        cfl_dt = self.dx / (self.c * np.sqrt(self.ndim)) * 0.5
        self.dt = cfl_dt
        self.n_steps = int(np.ceil(self.T / self.dt))
        self.dt = self.T / self.n_steps  # adjust for exact final time

        log.info(
            "WaveSimulator: c=%.2f, chi=%.2f, ndim=%d, Nx=%d, Nt=%d, dt=%.2e",
            self.c,
            self.chi,
            self.ndim,
            self.Nx,
            self.Nt,
            self.dt,
        )

    def _random_fourier_ic_1d(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random Fourier initial condition in 1D."""
        u0 = np.zeros(self.Nx, dtype=np.float64)
        for _ in range(self.n_modes):
            k = rng.integers(1, self.Nx // 4)
            amp = rng.uniform(0.1, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(2 * np.pi * k * self.x + phase)
        # Normalize to unit max amplitude
        u0 /= np.max(np.abs(u0)) + 1e-12
        return u0

    def _random_fourier_ic_2d(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random Fourier initial condition in 2D."""
        u0 = np.zeros((self.Nx, self.Nx), dtype=np.float64)
        for _ in range(self.n_modes):
            kx = rng.integers(1, self.Nx // 8)
            ky = rng.integers(1, self.Nx // 8)
            amp = rng.uniform(0.1, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(2 * np.pi * (kx * self.X + ky * self.Y) + phase)
        u0 /= np.max(np.abs(u0)) + 1e-12
        return u0

    def _laplacian_1d(self, u: np.ndarray) -> np.ndarray:
        """Periodic Laplacian in 1D via finite differences."""
        return (np.roll(u, 1) + np.roll(u, -1) - 2 * u) / self.dx**2

    def _laplacian_2d(self, u: np.ndarray) -> np.ndarray:
        """Periodic Laplacian in 2D via finite differences."""
        return (
            np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=1)
            + np.roll(u, -1, axis=1)
            - 4 * u
        ) / self.dx**2

    def generate_sample(self) -> np.ndarray:
        """
        Generate one sample of the wave/Klein-Gordon equation.

        Returns:
            np.ndarray: Solution array of shape (Nt, Nx) for 1D
                        or (Nt, Nx, Nx) for 2D, dtype float32.
        """
        rng = np.random.default_rng(self.seed)

        # Initial condition
        if self.ndim == 1:
            u0 = self._random_fourier_ic_1d(rng)
            laplacian = self._laplacian_1d
        else:
            u0 = self._random_fourier_ic_2d(rng)
            laplacian = self._laplacian_2d

        # Leapfrog integration
        u_prev = u0.copy()
        u_curr = u0.copy()  # du/dt = 0 at t=0

        # First half-step (Taylor expansion for du/dt=0)
        accel = self.c**2 * laplacian(u0) - self.chi**2 * u0
        u_curr = u0 + 0.5 * self.dt**2 * accel

        # Save schedule
        if self.ndim == 1:
            result = np.zeros((self.Nt, self.Nx), dtype=np.float32)
        else:
            result = np.zeros((self.Nt, self.Nx, self.Nx), dtype=np.float32)

        result[0] = u0.astype(np.float32)

        # Precompute which integration step maps to each t_save entry
        save_step_to_idx = {}
        for k in range(1, self.Nt):
            target_step = int(round(self.t_save[k] / self.dt))
            target_step = min(target_step, self.n_steps)
            save_step_to_idx[target_step] = k

        c2dt2 = self.c**2 * self.dt**2
        chi2dt2 = self.chi**2 * self.dt**2

        for step in range(1, self.n_steps + 1):
            # Save before advance: u_curr is at time step * dt
            if step in save_step_to_idx:
                result[save_step_to_idx[step]] = u_curr.astype(np.float32)

            lap = laplacian(u_curr)
            u_next = 2 * u_curr - u_prev + c2dt2 * lap - chi2dt2 * u_curr
            u_prev = u_curr
            u_curr = u_next

        return result


def analytical_solution_1d(
    x: np.ndarray,
    t: np.ndarray,
    u0: np.ndarray,
    c: float,
    chi: float = 0.0,
) -> np.ndarray:
    """
    Compute analytical solution for 1D wave/KG equation via FFT.

    The exact solution decomposes u0 into Fourier modes. Each mode k
    oscillates at frequency omega_k = sqrt(c^2 * (2*pi*k)^2 + chi^2).

    :param x: spatial grid (Nx,)
    :param t: time points (Nt,)
    :param u0: initial condition (Nx,)
    :param c: wave speed
    :param chi: mass parameter
    :return: solution array (Nt, Nx)
    """
    Nx = len(x)
    u0_hat = np.fft.fft(u0)
    k = np.fft.fftfreq(Nx, d=1.0 / Nx)  # wavenumbers

    omega = np.sqrt(c**2 * (2 * np.pi * k) ** 2 + chi**2 + 0j).real

    result = np.zeros((len(t), Nx), dtype=np.float64)
    for i, ti in enumerate(t):
        # du/dt(0) = 0 => only cosine term
        u_hat_t = u0_hat * np.cos(omega * ti)
        result[i] = np.fft.ifft(u_hat_t).real

    return result


def analytical_solution_2d(
    x: np.ndarray,
    t: np.ndarray,
    u0: np.ndarray,
    c: float,
    chi: float = 0.0,
) -> np.ndarray:
    """
    Compute analytical solution for 2D wave/KG equation via 2D FFT.

    Each Fourier mode (kx, ky) oscillates at frequency
    omega = sqrt(c^2 * (2*pi)^2 * (kx^2 + ky^2) + chi^2).

    :param x: 1D spatial grid (Nx,), same for both dimensions
    :param t: time points (Nt,)
    :param u0: initial condition (Nx, Nx)
    :param c: wave speed
    :param chi: mass parameter
    :return: solution array (Nt, Nx, Nx)
    """
    Nx = u0.shape[0]
    u0_hat = np.fft.fft2(u0)
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    ky = np.fft.fftfreq(Nx, d=1.0 / Nx)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    omega = np.sqrt(
        c**2 * (2 * np.pi) ** 2 * (KX**2 + KY**2) + chi**2 + 0j
    ).real

    result = np.zeros((len(t), Nx, Nx), dtype=np.float64)
    for i, ti in enumerate(t):
        # du/dt(0) = 0 => only cosine term
        u_hat_t = u0_hat * np.cos(omega * ti)
        result[i] = np.fft.ifft2(u_hat_t).real

    return result
