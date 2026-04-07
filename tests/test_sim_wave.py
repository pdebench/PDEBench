"""Tests for the 1D/2D wave equation and Klein-Gordon simulator."""

from __future__ import annotations

import numpy as np
import pytest

from pdebench.data_gen.src.sim_wave import WaveSimulator, analytical_solution_1d, analytical_solution_2d


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_wave_1d_output_shape():
    """generate_sample() for 1D should return (Nt, Nx) float32."""
    sim = WaveSimulator(c=1.0, chi=0.0, xdim=64, tdim=21, t=1.0, seed=0)
    result = sim.generate_sample()
    assert result.shape == (21, 64), f"Expected (21, 64), got {result.shape}"
    assert result.dtype == np.float32


def test_wave_2d_output_shape():
    """generate_sample() for 2D should return (Nt, Nx, Nx) float32."""
    sim = WaveSimulator(c=1.0, chi=0.0, xdim=32, tdim=11, t=0.5, ndim=2, seed=0)
    result = sim.generate_sample()
    assert result.shape == (11, 32, 32), f"Expected (11, 32, 32), got {result.shape}"
    assert result.dtype == np.float32


def test_wave_output_finite():
    """All output values should be finite (no NaN or Inf)."""
    sim = WaveSimulator(c=1.0, chi=0.0, xdim=64, tdim=21, t=1.0, seed=7)
    result = sim.generate_sample()
    assert np.isfinite(result).all(), "Output contains NaN or Inf"


def test_invalid_ndim_raises():
    """ndim=3 should raise ValueError."""
    with pytest.raises(ValueError, match="ndim must be 1 or 2"):
        WaveSimulator(ndim=3)


# ---------------------------------------------------------------------------
# Klein-Gordon (chi > 0)
# ---------------------------------------------------------------------------


def test_klein_gordon_1d_runs():
    """Klein-Gordon (chi=2.0) should run without error and return finite values."""
    sim = WaveSimulator(c=1.0, chi=2.0, xdim=64, tdim=11, t=0.5, seed=0)
    result = sim.generate_sample()
    assert result.shape == (11, 64)
    assert np.isfinite(result).all()


def test_klein_gordon_2d_runs():
    """Klein-Gordon 2D should run without error."""
    sim = WaveSimulator(c=1.0, chi=1.5, xdim=32, tdim=11, t=0.5, ndim=2, seed=0)
    result = sim.generate_sample()
    assert result.shape == (11, 32, 32)
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Accuracy: leapfrog vs analytical solution
# ---------------------------------------------------------------------------


def test_leapfrog_matches_analytical_single_mode():
    """
    Single k=1 cosine mode: leapfrog nRMSE vs analytical solution < 1%.

    For u0 = cos(2*pi*x) with du/dt=0, the exact solution is
        u(x,t) = cos(2*pi*x) * cos(2*pi*c*t).

    With Nx=256 and k=1, the FD dispersion error is < 0.02%, so the
    leapfrog and analytical solutions should agree to well within 1%.
    """
    sim = WaveSimulator(c=1.0, chi=0.0, xdim=256, tdim=21, t=0.5, seed=0)
    u0 = np.cos(2 * np.pi * sim.x)

    # Patch the IC generator to return our single-mode IC
    sim._random_fourier_ic_1d = lambda rng: u0.copy()  # noqa: ARG005

    numerical = sim.generate_sample()  # (21, 256), float32
    exact = analytical_solution_1d(sim.x, sim.t_save, u0, c=1.0, chi=0.0)

    u_range = exact.max() - exact.min()
    nrmse = np.sqrt(np.mean((numerical.astype(np.float64) - exact) ** 2)) / u_range
    assert nrmse < 0.01, f"nRMSE={nrmse:.4f} exceeds 1% tolerance"


def test_klein_gordon_leapfrog_matches_analytical():
    """
    Single k=1 cosine mode with chi=2: leapfrog nRMSE < 1%.

    Exact: u(x,t) = cos(2*pi*x) * cos(omega*t), omega = sqrt((2*pi)^2 + chi^2).
    """
    chi = 2.0
    sim = WaveSimulator(c=1.0, chi=chi, xdim=256, tdim=21, t=0.5, seed=0)
    u0 = np.cos(2 * np.pi * sim.x)

    sim._random_fourier_ic_1d = lambda rng: u0.copy()  # noqa: ARG005

    numerical = sim.generate_sample()
    exact = analytical_solution_1d(sim.x, sim.t_save, u0, c=1.0, chi=chi)

    u_range = exact.max() - exact.min()
    nrmse = np.sqrt(np.mean((numerical.astype(np.float64) - exact) ** 2)) / u_range
    assert nrmse < 0.01, f"KG nRMSE={nrmse:.4f} exceeds 1% tolerance"


# ---------------------------------------------------------------------------
# analytical_solution_1d sanity check
# ---------------------------------------------------------------------------


def test_analytical_solution_1d_at_t0():
    """Analytical solution at t=0 must equal u0 exactly."""
    Nx = 128
    x = np.linspace(0, 1, Nx, endpoint=False)
    u0 = np.sin(4 * np.pi * x) + 0.5 * np.cos(2 * np.pi * x)
    t = np.array([0.0, 0.25, 0.5])

    result = analytical_solution_1d(x, t, u0, c=1.0, chi=0.0)

    np.testing.assert_allclose(
        result[0], u0, atol=1e-12, err_msg="Analytical solution at t=0 != u0"
    )


def test_analytical_solution_kg_at_t0():
    """Analytical solution with chi>0 at t=0 must equal u0."""
    Nx = 64
    x = np.linspace(0, 1, Nx, endpoint=False)
    u0 = np.cos(2 * np.pi * x)
    t = np.array([0.0, 0.1])

    result = analytical_solution_1d(x, t, u0, c=1.0, chi=5.0)

    np.testing.assert_allclose(result[0], u0, atol=1e-12)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    """Same seed must produce bit-identical output across two calls."""
    sim_a = WaveSimulator(c=1.0, chi=0.0, xdim=32, tdim=11, t=0.5, seed=42)
    sim_b = WaveSimulator(c=1.0, chi=0.0, xdim=32, tdim=11, t=0.5, seed=42)
    np.testing.assert_array_equal(sim_a.generate_sample(), sim_b.generate_sample())


# ---------------------------------------------------------------------------
# 2D analytical solution
# ---------------------------------------------------------------------------


def test_wave_2d_matches_analytical():
    """
    2D leapfrog should match the 2D analytical solution to within 1% nRMSE.

    IC: u0(x, y) = cos(2*pi*x) * cos(2*pi*y), du/dt=0.
    Exact: u(x, y, t) = cos(2*pi*x) * cos(2*pi*y) * cos(2*pi*sqrt(2)*c*t).
    """
    Nx = 64
    sim = WaveSimulator(c=1.0, chi=0.0, xdim=Nx, tdim=11, t=0.3, ndim=2, seed=0)

    x1d = np.linspace(0, 1, Nx, endpoint=False)
    X, Y = np.meshgrid(x1d, x1d, indexing="ij")
    u0_2d = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    sim._random_fourier_ic_2d = lambda rng: u0_2d.copy()  # noqa: ARG005

    numerical = sim.generate_sample()  # (11, Nx, Nx) float32
    exact = analytical_solution_2d(x1d, sim.t_save, u0_2d, c=1.0, chi=0.0)

    u_range = exact.max() - exact.min()
    nrmse = np.sqrt(np.mean((numerical.astype(np.float64) - exact) ** 2)) / u_range
    assert nrmse < 0.01, f"2D nRMSE={nrmse:.4f} exceeds 1% tolerance"


def test_analytical_solution_2d_at_t0():
    """2D analytical solution at t=0 must equal u0 exactly."""
    Nx = 32
    x = np.linspace(0, 1, Nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = np.sin(2 * np.pi * X) + 0.3 * np.cos(4 * np.pi * Y)
    t = np.array([0.0, 0.5, 1.0])

    result = analytical_solution_2d(x, t, u0, c=1.0, chi=0.0)

    np.testing.assert_allclose(result[0], u0, atol=1e-12)


# ---------------------------------------------------------------------------
# Periodicity: single mode returns to IC after one period
# ---------------------------------------------------------------------------


def test_wave_periodicity_single_mode():
    """
    After one full period T = 1/c, a single k=1 cosine IC returns to itself.

    u(x, t) = cos(2*pi*x) * cos(2*pi*c*t) has period T = 1/c.
    The leapfrog nRMSE at t=T should be < 0.1% (O(dt^2) per step).
    """
    c = 1.0
    Nx = 256
    # tdim=3 saves t=0, t=T/2, t=T
    sim = WaveSimulator(c=c, chi=0.0, xdim=Nx, tdim=3, t=1.0 / c, seed=0)
    u0 = np.cos(2 * np.pi * sim.x)

    sim._random_fourier_ic_1d = lambda rng: u0.copy()  # noqa: ARG005

    result = sim.generate_sample()  # (3, Nx): t=0, t=T/2, t=T
    u_final = result[-1].astype(np.float64)

    u_range = u0.max() - u0.min()
    nrmse = np.sqrt(np.mean((u_final - u0) ** 2)) / u_range
    assert nrmse < 0.001, f"Periodicity nRMSE={nrmse:.6f} exceeds 0.1%"
