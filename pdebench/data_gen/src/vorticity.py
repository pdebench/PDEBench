r"""Generate vorticity field :math:`\boldsymbol{\omega} = \nabla \times \boldsymbol{v}` given
     velocity field :math:`\boldsymbol{v}` using numerical approximation.

Assuming the velocitiy field of shape [n, sx, sy, sz, 3] (5D) consists of a trajectory of equidistant cells,
the vorticity field is calculated by using spectral derivatives and the Fast Fourier Transform
such that :math:`\mathcal{F}\{\frac{\partial f}{\partial x}\ = i \omega \mathcal{F}\{f\}}`.

The code is inspired by
Brunton, S. L.,; Kutz, J. N. (2022). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control (2nd ed.).
and adapted to operate on 5D data.

This module provides the functions
  - compute_spectral_vorticity_np (numpy)
  - compute_spectral_vorticity_jnp (jax.numpy)

for approximating the vorticity field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def compute_spectral_vorticity_np(
    velocities: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    r"""Compute vorcitity field of a [n, sx, sy, sz, 3] field,
       where n denotes the number of timesteps, sx, sy, sz are the number
       of bins in x-, y-, z-direction.

    :param velocities: 5D Velocity field of shape [n, sx, sy, sz, 3].
    :param lx: length x-direction
    :type lx: float
    :param ly: length y-direction
    :type ly: float
    :param lz: length z-direction
    :type lz: float
    :type velocities: np.ndarray
    :raises ValueError: If `velocities` is not a 5D array or
        of shape [n, sx, sy, sz, 3].
    :return: Vorticity field :math:`\boldsymbol{\omega} \in \mathbb{R}^{n \times s_x \times s_y \times s_z \times 3}`
    :rtype: np.ndarray
    """

    if velocities.ndim != 5 or velocities.shape[-1] != 3:
        msg = "Expected 5D array of shape [n, sx, sy, sz, 3]!"
        raise ValueError(msg)

    dx = abs(dx)
    dy = abs(dy)
    dz = abs(dz)

    vx = velocities[..., 0]
    vy = velocities[..., 1]
    vz = velocities[..., 2]

    fxy = np.fft.fft(vx, axis=2)
    fyx = np.fft.fft(vy, axis=1)
    fxz = np.fft.fft(vx, axis=3)
    fzx = np.fft.fft(vz, axis=1)
    fyz = np.fft.fft(vy, axis=3)
    fzy = np.fft.fft(vz, axis=2)

    kappa_xy = 2.0 * np.pi * np.fft.fftfreq(n=fxy.shape[2], d=dy)
    kappa_yx = 2.0 * np.pi * np.fft.fftfreq(n=fyx.shape[1], d=dx)
    kappa_xz = 2.0 * np.pi * np.fft.fftfreq(n=fxz.shape[3], d=dz)
    kappa_zx = 2.0 * np.pi * np.fft.fftfreq(n=fzx.shape[1], d=dx)
    kappa_yz = 2.0 * np.pi * np.fft.fftfreq(n=fyz.shape[3], d=dz)
    kappa_zy = 2.0 * np.pi * np.fft.fftfreq(n=fzy.shape[2], d=dy)

    vxy = np.fft.ifft(1j * kappa_xy[None, None, :, None] * fxy, axis=2).real
    vyx = np.fft.ifft(1j * kappa_yx[None, :, None, None] * fyx, axis=1).real
    vxz = np.fft.ifft(1j * kappa_xz[None, None, None, :] * fxz, axis=3).real
    vzx = np.fft.ifft(1j * kappa_zx[None, :, None, None] * fzx, axis=1).real
    vyz = np.fft.ifft(1j * kappa_yz[None, None, None, :] * fyz, axis=3).real
    vzy = np.fft.ifft(1j * kappa_zy[None, None, :, None] * fzy, axis=2).real

    omega_x = vzy - vyz
    omega_y = vxz - vzx
    omega_z = vyx - vxy

    return np.concatenate(
        [omega_x[..., None], omega_y[..., None], omega_z[..., None]], axis=-1
    )


@jax.jit  # type: ignore[misc]
def compute_spectral_vorticity_jnp(
    velocities: jnp.ndarray, dx: float, dy: float, dz: float
) -> jnp.ndarray:
    r"""Compute vorcitity field of a [n, sx, sy, sz, 3] field,
       where n denotes the number of timesteps, sx, sy, sz are the number
       of bins in x-, y-, z-direction. In this case computations are performed on GPU

    :param velocities: 5D Velocity field of shape [n, sx, sy, sz, 3].
    :param lx: length x-direction
    :type lx: float
    :param ly: length y-direction
    :type ly: float
    :param lz: length z-direction
    :type lz: float
    :type velocities: np.ndarray
    :raises ValueError: If `velocities` is not a 5D array or
        of shape [n, sx, sy, sz, 3].
    :return: Vorticity field :math:`\boldsymbol{\omega} \in \mathbb{R}^{n \times s_x \times s_y \times s_z \times 3}`
    :rtype: np.ndarray
    """

    if velocities.ndim != 5 or velocities.shape[-1] != 3:
        msg = "Expected 5D array of shape [n, sx, sy, sz, 3]!"
        raise ValueError(msg)

    dx = abs(dx)
    dy = abs(dy)
    dz = abs(dz)

    vx = velocities[..., 0]
    vy = velocities[..., 1]
    vz = velocities[..., 2]

    fxy = jnp.fft.fft(vx, axis=2)
    fyx = jnp.fft.fft(vy, axis=1)
    fxz = jnp.fft.fft(vx, axis=3)
    fzx = jnp.fft.fft(vz, axis=1)
    fyz = jnp.fft.fft(vy, axis=3)
    fzy = jnp.fft.fft(vz, axis=2)

    kappa_xy = 2.0 * jnp.pi * jnp.fft.fftfreq(n=fxy.shape[2], d=dy)
    kappa_yx = 2.0 * jnp.pi * jnp.fft.fftfreq(n=fyx.shape[1], d=dx)
    kappa_xz = 2.0 * jnp.pi * jnp.fft.fftfreq(n=fxz.shape[3], d=dz)
    kappa_zx = 2.0 * jnp.pi * jnp.fft.fftfreq(n=fzx.shape[1], d=dx)
    kappa_yz = 2.0 * jnp.pi * jnp.fft.fftfreq(n=fyz.shape[3], d=dz)
    kappa_zy = 2.0 * jnp.pi * jnp.fft.fftfreq(n=fzy.shape[2], d=dy)

    vxy = jnp.fft.ifft(1j * kappa_xy[None, None, :, None] * fxy, axis=2).real
    vyx = jnp.fft.ifft(1j * kappa_yx[None, :, None, None] * fyx, axis=1).real
    vxz = jnp.fft.ifft(1j * kappa_xz[None, None, None, :] * fxz, axis=3).real
    vzx = jnp.fft.ifft(1j * kappa_zx[None, :, None, None] * fzx, axis=1).real
    vyz = jnp.fft.ifft(1j * kappa_yz[None, None, None, :] * fyz, axis=3).real
    vzy = jnp.fft.ifft(1j * kappa_zy[None, None, :, None] * fzy, axis=2).real

    omega_x = vzy - vyz
    omega_y = vxz - vzx
    omega_z = vyx - vxy

    return jnp.concatenate(
        [omega_x[..., None], omega_y[..., None], omega_z[..., None]], axis=-1
    )
