from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from pdebench.data_gen.src.vorticity import (
    compute_spectral_vorticity_jnp,
    compute_spectral_vorticity_np,
)


@pytest.fixture
def generate_random_spectral_velvor() -> tuple[np.ndarray, np.ndarray]:
    """Generate random 5D velocity- and corresponding vorticity field

    :return: Velocity- and vorticity field
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    generator = np.random.default_rng(seed=None)
    vel = generator.uniform(size=(10, 16, 32, 32, 3))
    vx = vel[..., 0]
    vy = vel[..., 1]
    vz = vel[..., 2]

    fxy = np.fft.fft(vx, axis=2)
    fxz = np.fft.fft(vx, axis=3)
    fyx = np.fft.fft(vy, axis=1)
    fyz = np.fft.fft(vy, axis=3)
    fzx = np.fft.fft(vz, axis=1)
    fzy = np.fft.fft(vz, axis=2)

    kappa_xy = 2.0 * np.pi * np.fft.fftfreq(vel.shape[2], 1.0 / vel.shape[2])
    kappa_xz = 2.0 * np.pi * np.fft.fftfreq(vel.shape[3], 1.0 / vel.shape[3])
    kappa_yx = 2.0 * np.pi * np.fft.fftfreq(vel.shape[1], 1.0 / vel.shape[1])
    kappa_yz = 2.0 * np.pi * np.fft.fftfreq(vel.shape[3], 1.0 / vel.shape[3])
    kappa_zx = 2.0 * np.pi * np.fft.fftfreq(vel.shape[1], 1.0 / vel.shape[1])
    kappa_zy = 2.0 * np.pi * np.fft.fftfreq(vel.shape[2], 1.0 / vel.shape[2])

    vxy = np.fft.ifft(1j * kappa_xy[None, None, :, None] * fxy, axis=2).real
    vyx = np.fft.ifft(1j * kappa_yx[None, :, None, None] * fyx, axis=1).real
    vxz = np.fft.ifft(1j * kappa_xz[None, None, None, :] * fxz, axis=3).real
    vzx = np.fft.ifft(1j * kappa_zx[None, :, None, None] * fzx, axis=1).real
    vyz = np.fft.ifft(1j * kappa_yz[None, None, None, :] * fyz, axis=3).real
    vzy = np.fft.ifft(1j * kappa_zy[None, None, :, None] * fzy, axis=2).real

    omegax = vzy - vyz
    omegay = vxz - vzx
    omegaz = vyx - vxy

    omega = np.concatenate(
        [omegax[..., None], omegay[..., None], omegaz[..., None]], axis=-1
    )

    return vel, omega


def test_vorticity_np(generate_random_spectral_velvor) -> None:
    """Test approximated vorticity by spectral derivation"""
    vel, vort = generate_random_spectral_velvor
    dx = 1.0 / vel.shape[1]
    dy = 1.0 / vel.shape[2]
    dz = 1.0 / vel.shape[3]

    vort_np = compute_spectral_vorticity_np(vel, dx, dy, dz)
    np.testing.assert_almost_equal(vort_np, vort)


def test_vorticity_jnp(generate_random_spectral_velvor) -> None:
    """Test approximated vorticity by spectral derivation"""
    vel, vort = generate_random_spectral_velvor
    dx = 1.0 / vel.shape[1]
    dy = 1.0 / vel.shape[2]
    dz = 1.0 / vel.shape[3]

    vort_jnp = compute_spectral_vorticity_jnp(jnp.array(vel), dx, dy, dz)
    np.testing.assert_almost_equal(np.array(vort_jnp), vort, decimal=4)
