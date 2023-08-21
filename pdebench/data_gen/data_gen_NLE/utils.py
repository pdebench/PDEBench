#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math as mt
import jax
import numpy as np
import jax.numpy as jnp
from jax import random, jit, nn, lax, vmap, scipy
from functools import partial

# if double precision
#from jax.config import config
#config.update("jax_enable_x64", True)


def init(xc, mode='sin', u0=1., du=0.1):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    modes = ['sin', 'sinsin', 'Gaussian', 'react', 'possin']
    assert mode in modes, 'mode is not defined!!'
    if mode == 'sin':  # sinusoidal wave
        u = u0 * jnp.sin((xc + 1.) * jnp.pi)
    elif mode == 'sinsin':  # sinusoidal wave
        u = jnp.sin((xc + 1.) * jnp.pi) + du * jnp.sin((xc + 1.) * jnp.pi*8.)
    elif mode == 'Gaussian':  # for diffusion check
        t0 = 0.01
        u = jnp.exp(-xc**2*jnp.pi/(4.*t0))/jnp.sqrt(2.*t0)
    elif mode == 'react':  # for reaction-diffusion eq.
        logu = - 0.5*(xc - jnp.pi)**2/(0.25*jnp.pi)**2
        u = jnp.exp(logu)
    elif mode == 'possin':  # sinusoidal wave
        u = u0 * jnp.abs(jnp.sin((xc + 1.) * jnp.pi))
    return u


@partial(jit, static_argnums=(1, 2, 3, 4))
def init_multi(xc, numbers=10000, k_tot=8, init_key=2022, num_choise_k=2, if_norm=False):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """

    def _pass(carry):
        return carry

    def select_A(carry):
        def _func(carry):
            carry = jnp.abs(carry)
            return carry

        cond, value = carry
        value = lax.cond(cond == 1, _func, _pass, value)
        return cond, value

    def select_W(carry):
        def _window(carry):
            xx, val, xL, xR, trns = carry
            val = 0.5 * (jnp.tanh((xx - xL) / trns) - jnp.tanh((xx - xR) / trns))
            return xx, val, xL, xR, trns

        cond, value, xx, xL, xR, trns = carry

        carry = xx, value, xL, xR, trns
        xx, value, xL, xR, trns = lax.cond(cond == 1, _window, _pass, carry)
        return cond, value, xx, xL, xR, trns

    def normalize(carry):
        def _norm(carry):
            u = carry
            u -= jnp.min(u, axis=1, keepdims=True)  # positive value
            u /= jnp.max(u, axis=1, keepdims=True)  # normalize
            return u

        cond, u = carry
        u = lax.cond(cond==True, _norm, _pass, u)
        return cond, u

    key = random.PRNGKey(init_key)

    selected = random.randint(key, shape=[numbers, num_choise_k], minval=0, maxval=k_tot)
    selected = nn.one_hot(selected, k_tot, dtype=int).sum(axis=1)
    kk = jnp.pi * 2. * jnp.arange(1, k_tot + 1) * selected / (xc[-1] - xc[0])
    amp = random.uniform(key, shape=[numbers, k_tot, 1])

    key, subkey = random.split(key)

    phs = 2. * jnp.pi * random.uniform(key, shape=[numbers, k_tot, 1])
    _u = amp * jnp.sin(kk[:, :, jnp.newaxis] * xc[jnp.newaxis, jnp.newaxis, :] + phs)
    _u = jnp.sum(_u, axis=1)

    # perform absolute value function
    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=([numbers]))
    carry = (cond, _u)

    cond, _u = vmap(select_A, 0, 0)(carry)
    sgn = random.choice(key, a=jnp.array([1, -1]), shape=([numbers, 1]))
    _u *= sgn  # random flip of signature

    # perform window function
    key, subkey = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=([numbers]))
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)
    mask = jnp.ones_like(_xc)
    xL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    xR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    trns = 0.01 * jnp.ones_like(cond)
    carry = cond, mask, _xc, xL, xR, trns
    cond, mask, _xc, xL, xR, trns = vmap(select_W, 0, 0)(carry)

    _u *= mask

    carry = if_norm, _u
    _, _u = normalize(carry)  # normalize value between [0, 1] for reaction-diffusion eq.

    return _u

def init_multi_2DRand(xc, yc, numbers=10000, init_key=2022, k_tot=4, duMx = 1.e1):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    def _pass(carry):
        return carry

    def select_W(carry):
        def _window(carry):
            xx, yy, val, xL, xR, yL, yR, trns = carry
            x_win = 0.5 * (jnp.tanh((xx - xL) / trns) - jnp.tanh((xx - xR) / trns))
            y_win = 0.5 * (jnp.tanh((yy - yL) / trns) - jnp.tanh((yy - yR) / trns))
            val = x_win[:, None] * y_win[None, :]
            return xx, yy, val, xL, xR, yL, yR, trns

        cond, value, xx, yy, xL, xR, yL, yR, trns = carry

        carry = xx, yy, value, xL, xR, yL, yR, trns
        xx, yy, value, xL, xR, yL, yR, trns = lax.cond(cond == 1, _window, _pass, carry)
        return cond, value, xx, xL, xR, trns

    def __create_2DRand_init(u0, delu):
        nx, ny = xc.shape[0], yc.shape[0]

        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]

        qLx = dx * nx
        qLy = dy * ny

        ## random field
        u = jnp.zeros([nx, ny])

        key = random.PRNGKey(init_key)
        kx0 = jnp.pi * 2. / qLx
        ky0 = jnp.pi * 2. / qLy

        for j in range(-k_tot, k_tot + 1):
            ky = ky0 * j  # from 1 to k_tot
            for i in range(-k_tot, k_tot + 1):
                kx = kx0 * i  # from 1 to k_tot
                if i * j == 0:  # avoiding uniform velocity
                    continue
                # random phase
                key, subkey = random.split(key)
                phs = 2. * jnp.pi * random.uniform(key, shape=[1])  # (vi, k)

                uk = 1. / jnp.sqrt(jnp.sqrt(kx ** 2 + ky ** 2))
                kdx = kx * xc[:, None] + ky * yc[None, :]
                u += uk * jnp.sin(kdx + phs)

        # renormalize total velocity
        u = u0 + delu * u / jnp.abs(u).mean()

        return u

    key = random.PRNGKey(init_key)
    u0 = random.uniform(key, shape=([numbers, 1]), minval=1.e-1, maxval=duMx)
    key, subkey = random.split(key)
    delu = random.uniform(key, shape=([numbers, 1]), minval=1.e-2, maxval=0.5)
    u = jax.vmap(__create_2DRand_init, axis_name='i')(u0, delu)

    # perform window function
    key, subkey = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.5, 0.5]), shape=([numbers]))
    mask = jnp.ones([numbers, xc.shape[0], yc.shape[0]])
    xL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    xR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    key, subkey = random.split(key)
    yL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    yR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)
    _yc = jnp.repeat(xc[None, :], numbers, axis=0)
    trns = 0.01 * jnp.ones_like(cond)
    carry = cond, mask, _xc, _yc, xL, xR, yL, yR, trns
    cond, mask, _xc, xL, xR, trns = vmap(select_W, 0, 0)(carry)

    u = u * mask
    u = u + u0[:,:,None] * (1. - mask)

    return u

def init_HD(u, xc, yc, zc, mode='shocktube1', direc='x', init_key=2022,
            M0=0.1, dk=1, gamma=.1666666667):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    print(mode)
    modes = ['shocktube0','shocktube1','shocktube2','shocktube3','shocktube4','shocktube5','shocktube6','shocktube7',
             '2D-shock', 'OTVortex', 'KHI', 'turbulence', 'sound_wave', 'c_discon', 'BlastWave']
    assert mode in modes, 'mode is not defined!!'

    _, nx, ny, nz = u.shape

    if mode[:-1] == 'shocktube':  #  shock tube

        if direc == 'x':
            iX, iY, iZ = 1, 2, 3
            Ncell = nx
            _u = jnp.zeros_like(u)
        elif direc == 'y':
            iX, iY, iZ = 2, 3, 1
            Ncell = ny
            _u = jnp.transpose(u, (0, 2, 3, 1))
        if direc == 'z':
            iX, iY, iZ = 3, 1, 2
            Ncell = nz
            _u = jnp.transpose(u, (0, 3, 1, 2))

        if mode[-1] == '0':  # test 0  for viscosity
            nx0 = int(0.5*Ncell)
            uL = [1., 0.75, 0.2, -0.3, 1.]
            uR = [0.125, 0., 0.1, 0.9, 0.1]
        elif mode[-1] == '1':  # test 1
            nx0 = int(0.3*Ncell)
            uL = [1., 0.75, 0., 0., 1.]
            uR = [0.125, 0., 0., 0., 0.1]
        elif mode[-1] == '2':  # test 2
            nx0 = int(0.5*Ncell)
            uL = [1., -2., 0., 0., 0.4]
            uR = [1., 2., 0., 0., 0.4]
        elif mode[-1] == '3':  # test 3
            nx0 = int(0.5*Ncell)
            uL = [1., 0., 0., 0., 1.e3]
            uR = [1., 0., 0., 0., 0.01]
        elif mode[-1] == '4':  # test 4
            nx0 = int(0.4*Ncell)
            uL = [5.99924, 19.5975, 0., 0., 460.894]
            uR = [5.99242, -6.19633, 0., 0., 46.095]
        elif mode[-1] == '5':  # test 5
            nx0 = int(0.8 * Ncell)
            uL = [1., -19.59745, 0., 0., 1.e3]
            uR = [1., -19.59745, 0., 0., 0.01]
        elif mode[-1] == '6':  # test 6
            nx0 = int(0.5 * Ncell)
            uL = [1.4, 0., 0., 0., 1.]
            uR = [1., 0., 0., 0., 1.]
        elif mode[-1] == '7':  # test 7
            nx0 = int(0.5 * Ncell)
            uL = [1.4, 0.1, 0., 0., 1.]
            uR = [1., 0.1, 0., 0., 1.]

        # left
        _u = _u.at[0, :nx0].set(uL[0])
        _u = _u.at[iX, :nx0].set(uL[1])
        _u = _u.at[iY, :nx0].set(uL[2])
        _u = _u.at[iZ, :nx0].set(uL[3])
        _u = _u.at[4, :nx0].set(uL[4])
        # right
        _u = _u.at[0, nx0:].set(uR[0])
        _u = _u.at[iX, nx0:].set(uR[1])
        _u = _u.at[iY, nx0:].set(uR[2])
        _u = _u.at[iZ, nx0:].set(uR[3])
        _u = _u.at[4, nx0:].set(uR[4])

        if direc == 'x':
            u = _u
        elif direc == 'y':
            u = jnp.transpose(_u, (0, 3, 1, 2))
        elif direc == 'z':
            u = jnp.transpose(_u, (0, 2, 3, 1))
    elif mode == '2D-shock':  #  shock tube
        u1 = [0.5, 0., 0., 0., 0.1]
        u2 = [0.1, 0., 1., 0., 1.]
        u3 = [0.1, 1., 0., 0., 1.]
        u4 = [0.1, 0., 0., 0., 0.01]

        # left-bottom
        u = u.at[0, :nx//2, :ny//2].set(u1[0])
        u = u.at[1, :nx//2, :ny//2].set(u1[1])
        u = u.at[2, :nx//2, :ny//2].set(u1[2])
        u = u.at[3, :nx//2, :ny//2].set(u1[3])
        u = u.at[4, :nx//2, :ny//2].set(u1[4])
        # right-bottom
        u = u.at[0, nx//2:, :ny//2].set(u2[0])
        u = u.at[1, nx//2:, :ny//2].set(u2[1])
        u = u.at[2, nx//2:, :ny//2].set(u2[2])
        u = u.at[3, nx//2:, :ny//2].set(u2[3])
        u = u.at[4, nx//2:, :ny//2].set(u2[4])
        # left-top
        u = u.at[0, :nx//2, ny//2:].set(u3[0])
        u = u.at[1, :nx//2, ny//2:].set(u3[1])
        u = u.at[2, :nx//2, ny//2:].set(u3[2])
        u = u.at[3, :nx//2, ny//2:].set(u3[3])
        u = u.at[4, :nx//2, ny//2:].set(u3[4])
        # right-top
        u = u.at[0, nx//2:, ny//2:].set(u4[0])
        u = u.at[1, nx//2:, ny//2:].set(u4[1])
        u = u.at[2, nx//2:, ny//2:].set(u4[2])
        u = u.at[3, nx//2:, ny//2:].set(u4[3])
        u = u.at[4, nx//2:, ny//2:].set(u4[4])

    elif mode == 'OTVortex':  # shock tube
        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        qLx = dx * xc.shape[0]
        qLy = dy * yc.shape[0]
        _xc = jnp.zeros([xc.shape[0] + 4])
        _yc = jnp.zeros([yc.shape[0] + 4])
        _xc = _xc.at[2:-2].set(xc)
        _yc = _yc.at[2:-2].set(yc)
        _xc = _xc.at[:2].set(jnp.array([-2 * dx, -dx]))
        _yc = _yc.at[:2].set(jnp.array([-2 * dy, -dy]))
        _xc = _xc.at[-2:].set(jnp.array([xc[-1] + dx, xc[-1] + 2. * dx]))
        _yc = _yc.at[-2:].set(jnp.array([yc[-1] + dy, yc[-1] + 2. * dy]))

        u = u.at[0].add(gamma ** 2)
        u = u.at[1].set(- jnp.sin(2. * jnp.pi * _yc[None, :, None] / qLy))
        u = u.at[2].set(jnp.sin(2. * jnp.pi * _xc[:, None, None] / qLx))
        u = u.at[3].add(0.)
        u = u.at[4].add(gamma)

    elif mode == 'KHI':  # Kelvin-Helmholtz instability
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        #gamma = 1.666666666666667
        #k = 1.  # moved to the external input
        d0_u = 2./(dk + 1.)
        d0_d = dk * d0_u
        d0 = 0.5 * (d0_u + d0_d)
        #M0 = 0.1  #  Mach number  # moved to external input
        ux = 1.
        cs = ux/M0
        #ux = 0.1 * cs # << cs
        p0 = cs**2 * d0/gamma
        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        qLx = dx * nx
        qLy = dy * ny
        kk = 4.  # wave number
        kx = kk * 2. * jnp.pi / qLx
        dl = 5.e-3 * qLy

        bound = 0.5 * qLy + dl * jnp.sin(kx * xc)  # assuming yL = 0

        vx = jnp.zeros([nx, ny, nz])
        dd = jnp.zeros([nx, ny, nz])
        for i in range(nx):
            _vx = jnp.where(yc > bound[i], ux, -ux)
            _dd = jnp.where(yc > bound[i], d0_u, d0_d)
            vx = vx.at[i, :, :].set(_vx[:, None])
            dd = dd.at[i, :, :].set(_dd[:, None])

        u = u.at[0, 2:-2, 2:-2, 2:-2].set(dd)
        u = u.at[1, 2:-2, 2:-2, 2:-2].set(vx)
        u = u.at[2].set(0.)
        u = u.at[3].add(0.)
        u = u.at[4].add(p0)

    elif mode == 'turbulence':  # 3D decaying turbulence

        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        d0 = 1.
        cs = 1./M0
        u0 = 1.  # fixed
        p0 = cs ** 2 * d0 / gamma

        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        dz = zc[1] - zc[0]
        qLx = dx * nx
        qLy = dy * ny
        qLz = dz * nz

        ## random velocity field
        k_tot = 3
        vx, vy, vz = np.zeros([nx, ny, nz]), np.zeros([nx, ny, nz]), np.zeros([nx, ny, nz])

        key = random.PRNGKey(init_key)

        kx0 = jnp.pi * 2. / qLx
        ky0 = jnp.pi * 2. / qLy
        kz0 = jnp.pi * 2. / qLz

        for k in range(-k_tot, k_tot + 1):
            kz = kz0 * k  # from 1 to k_tot
            for j in range(-k_tot, k_tot + 1):
                ky = ky0 * j  # from 1 to k_tot
                for i in range(-k_tot, k_tot + 1):
                    kx = kx0 * i  # from 1 to k_tot
                    if i * j * k == 0:  # avoiding uniform velocity
                        continue
                    # random phase
                    key, subkey = random.split(key)
                    phs = 2. * jnp.pi * random.uniform(key, shape=[3])  # (vi, k)

                    uk = 1./jnp.sqrt(kx**2 + ky**2 + kz**2)
                    kdx = kx * xc[:,None,None] + ky * yc[None,:,None] + kz * zc[None,None,:]
                    vx += uk * jnp.sin(kdx + phs[0])
                    vy += uk * jnp.sin(kdx + phs[1])
                    vz += uk * jnp.sin(kdx + phs[2])

        del(kdx, uk, phs)

        # Helmholtz decomposition to subtract expansion: k.vk
        dfx, dfy, dfz = 1./qLx, 1./qLy, 1./qLz
        fx = dfx * (np.arange(nx) - 1. - nx//2)
        fy = dfy * (np.arange(ny) - 1. - ny//2)
        fz = dfz * (np.arange(nz) - 1. - nz//2)

        vkx = np.fft.fftn(vx) * dx * dy * dz
        vky = np.fft.fftn(vy) * dx * dy * dz
        vkz = np.fft.fftn(vz) * dx * dy * dz

        # shift to kxi=0 is at the center
        vkx = np.fft.fftshift(vkx)
        vky = np.fft.fftshift(vky)
        vkz = np.fft.fftshift(vkz)

        #for k in range(nz):
        #    for j in range(ny):
        #        for i in range(nx):
        #            ff = (fx[i]**2 + fy[j]**2 + fz[k]**2)
        #            fi = np.where(ff > 1.e-8, 1./ff, 0.)
        #            # subtract expansion k.vk
        #            fdv = fx[i] * vkx[i, j, k] + fy[j] * vky[i, j, k] + fz[k] * vkz[i, j, k]
        #            vkx -= fdv * fx[i] * fi
        #            vky -= fdv * fy[j] * fi
        #            vkz -= fdv * fz[k] * fi

        fi = fx[:,None,None]**2 + fy[None,:,None]**2 + fz[None,None,:]**2
        fi = np.where(fi > 1.e-8, 1./fi, 0.)

        fdv = (fx[:,None,None] * vkx + fy[None,:,None] * vky + fz[None,None,:] * vkz) * fi
        vkx -= fdv * fx[:,None,None]
        vky -= fdv * fy[None,:,None]
        vkz -= fdv * fz[None,None,:]
        del(fi, fdv)

        # shift back to original order
        vkx = np.fft.ifftshift(vkx)
        vky = np.fft.ifftshift(vky)
        vkz = np.fft.ifftshift(vkz)

        # inverse FFT
        vx = np.fft.ifftn(vkx).real * dfx * dfy * dfz
        vy = np.fft.ifftn(vky).real * dfx * dfy * dfz
        vz = np.fft.ifftn(vkz).real * dfx * dfy * dfz

        # renormalize total velocity
        vtot = np.sqrt(vx**2 + vy**2 + vz**2).mean()
        vx *= u0 / vtot
        vy *= u0 / vtot
        vz *= u0 / vtot

        u = u.at[0].set(d0)
        u = u.at[1,2:-2,2:-2,2:-2].set(jnp.array(vx))
        u = u.at[2,2:-2,2:-2,2:-2].set(jnp.array(vy))
        u = u.at[3,2:-2,2:-2,2:-2].add(jnp.array(vz))
        u = u.at[4].add(p0)

    elif mode == 'BlastWave':  # Kelvin-Helmholtz instability
        """ Stone Gardiner 2009 without B """
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        db = 1.
        pb = 0.1

        pc = 1.e2  # central region

        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        dz = zc[1] - zc[0]
        qLx = dx * nx
        qLy = dy * ny
        qLz = dz * nz
        qL = (qLx + qLy + qLz)/3.

        #p0 = jnp.ones([nx, ny, nz]) * pb
        RR = jnp.sqrt((xc[:,None,None] - xc[nx//2])**2
                    + (yc[None,:,None] - yc[ny//2])**2
                    + (zc[None,None,:] - zc[nz//2])**2)
        p0 = jnp.where(RR > 0.05 * qL, pb, pc)
        #for k in range(nz):
        #    for j in range(ny):
        #        for i in range(nx):
        #            RR = jnp.sqrt((xc[i] - 0.5 * qLx)**2 + (yc[j] - 0.5 * qLy)**2 + (zc[k] - 0.5 * qLz)**2)
        #            if RR < 0.1 * qL:
        #                p0 = p0.at[i,j,k].set(pc)

        u = u.at[0].set(db)
        u = u.at[1].set(0.)
        u = u.at[2].set(0.)
        u = u.at[3].set(0.)
        u = u.at[4, 2:-2, 2:-2, 2:-2].set(p0)

    elif mode == 'sound_wave':  #  sound wave
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        gamma = 1.666666666666667
        d0 = 1.
        cs = 2.
        p0 = cs**2 * d0/gamma
        if direc == 'x':
            iX, iY, iZ = 1, 2, 3
            XC = xc
            qL = (xc[1] - xc[0]) * nx
            _u = jnp.zeros_like(u)
        elif direc == 'y':
            iX, iY, iZ = 2, 3, 1
            XC = yc
            qL = (yc[1] - yc[0]) * ny
            _u = jnp.transpose(u, (0, 2, 3, 1))
        if direc == 'z':
            iX, iY, iZ = 3, 1, 2
            XC = zc
            qL = (zc[1] - zc[0]) * nz
            _u = jnp.transpose(u, (0, 3, 1, 2))

        kk = 2. * jnp.pi / qL
        _u = _u.at[0,2:-2].set(d0 * (1. + 1.e-3 * jnp.sin(kk * XC[:, None, None])))
        _u = _u.at[iX].set((_u[0] - d0) * cs /d0)
        _u = _u.at[4].set(p0 + cs**2 * (_u[0] - d0) )

        if direc == 'x':
            u = _u
        elif direc == 'y':
            u = jnp.transpose(_u, (0, 3, 1, 2))
        elif direc == 'z':
            u = jnp.transpose(_u, (0, 2, 3, 1))

    elif mode == 'c_discon':  #  tangent discontinuity
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        d0 = 1.
        p0 = 1.
        vy0 = 0.1
        if direc == 'x':
            iX, iY, iZ = 1, 2, 3
            XC = xc
            qL = (xc[1] - xc[0]) * nx
            _u = jnp.zeros_like(u)
        elif direc == 'y':
            iX, iY, iZ = 2, 3, 1
            XC = yc
            qL = (yc[1] - yc[0]) * ny
            _u = jnp.transpose(u, (0, 2, 3, 1))
        if direc == 'z':
            iX, iY, iZ = 3, 1, 2
            XC = zc
            qL = (zc[1] - zc[0]) * nz
            _u = jnp.transpose(u, (0, 3, 1, 2))

        _u = _u.at[0].set(d0)
        _u = _u.at[iY, 2:-2].set(vy0 * scipy.special.erf(0.5 * XC[:, None, None] / jnp.sqrt(0.1)))
        _u = _u.at[4].set(p0)

        if direc == 'x':
            u = _u
        elif direc == 'y':
            u = jnp.transpose(_u, (0, 3, 1, 2))
        elif direc == 'z':
            u = jnp.transpose(_u, (0, 2, 3, 1))

    return u

@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def init_multi_HD(xc, yc, zc, numbers=10000, k_tot=10, init_key=2022, num_choise_k=2,
                  if_renorm=False, umax=1.e4, umin=1.e-8):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """

    def _pass(carry):
        return carry

    def select_A(carry):
        def _func(carry):
            carry = jnp.abs(carry)
            return carry

        cond, value = carry
        value = lax.cond(cond == 1, _func, _pass, value)
        return cond, value

    def select_W(carry):
        def _window(carry):
            xx, val, xL, xR, trns = carry
            val = 0.5 * (jnp.tanh((xx - xL) / trns) - jnp.tanh((xx - xR) / trns))
            return xx, val, xL, xR, trns

        cond, value, xx, xL, xR, trns = carry

        carry = xx, value, xL, xR, trns
        xx, value, xL, xR, trns = lax.cond(cond == 1, _window, _pass, carry)
        return cond, value, xx, xL, xR, trns

    def renormalize(carry):
        def _norm(carry):
            u, key = carry
            u -= jnp.min(u, axis=1, keepdims=True)  # positive value
            u /= jnp.max(u, axis=1, keepdims=True)  # normalize

            key, subkey = random.split(key)
            m_val = random.uniform(key, shape=[numbers], minval=mt.log(umin), maxval=mt.log(umax))
            m_val = jnp.exp(m_val)
            key, subkey = random.split(key)
            b_val = random.uniform(key, shape=[numbers], minval=mt.log(umin), maxval=mt.log(umax))
            b_val = jnp.exp(b_val)
            return u * m_val[:, None] + b_val[:, None], key

        cond, u, key = carry
        carry = u, key
        u, key = lax.cond(cond==True, _norm, _pass, carry)
        return cond, u, key

    assert yc.shape[0] == 1 and zc.shape[0] == 1, 'ny and nz is assumed to be 1!!'
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    key = random.PRNGKey(init_key)

    selected = random.randint(key, shape=[numbers, num_choise_k], minval=0, maxval=k_tot)
    selected = nn.one_hot(selected, k_tot, dtype=int).sum(axis=1)
    kk = jnp.pi * 2. * jnp.arange(1, k_tot + 1) * selected / (xc[-1] - xc[0])
    amp = random.uniform(key, shape=[numbers, k_tot, 1])

    key, subkey = random.split(key)

    phs = 2. * jnp.pi * random.uniform(key, shape=[numbers, k_tot, 1])
    _u = amp * jnp.sin(kk[:, :, jnp.newaxis] * xc[jnp.newaxis, jnp.newaxis, :] + phs)
    _u = jnp.sum(_u, axis=1)

    # perform absolute value function
    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=([numbers]))
    carry = (cond, _u)

    cond, _u = vmap(select_A, 0, 0)(carry)
    sgn = random.choice(key, a=jnp.array([1, -1]), shape=([numbers, 1]))
    _u *= sgn  # random flip of signature

    # perform window function
    key, subkey = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.5, 0.5]), shape=([numbers]))
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)
    mask = jnp.ones_like(_xc)
    xL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    xR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    trns = 0.01 * jnp.ones_like(cond)
    carry = cond, mask, _xc, xL, xR, trns
    cond, mask, _xc, xL, xR, trns = vmap(select_W, 0, 0)(carry)

    _u *= mask

    carry = if_renorm, _u, key
    _, _u, _ = renormalize(carry)  # renormalize value between a given values

    return _u[...,None,None]

#@partial(jit, static_argnums=(3, 4, 5, 6))
def init_multi_HD_shock(xc, yc, zc, numbers=10000, init_key=2022, umax=1.e4, umin=1.e-8):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    assert yc.shape[0] == 1 and zc.shape[0] == 1, 'ny and nz is assumed to be 1!!'
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    def select_var(carry):
        def _func(carry):
            vmin, vmax = carry
            return jnp.log(vmin), jnp.log(vmax)

        def _pass(carry):
            return carry

        vmin, vmax = carry
        vmin, vmax = lax.cond(vmin > 0., _func, _pass, carry)
        return vmin, vmax

    nx = xc.shape[0]

    carry = umin, umax
    u_min, u_max = select_var(carry)

    key = random.PRNGKey(init_key)
    QLs = random.uniform(key, shape=([numbers, 1]), minval=u_min, maxval=u_max)
    QLs = jnp.exp(QLs)
    key, subkey = random.split(key)
    QRs = random.uniform(key, shape=([numbers, 1]), minval=u_min, maxval=u_max)
    QRs = jnp.exp(QRs)

    nx0s = nx * random.uniform(key, shape=([numbers, 1]), minval=0.25, maxval=0.75)
    nx0s = nx0s.astype(int)

    u = jnp.arange(xc.shape[0])
    u = jnp.tile(u, (numbers, 1))

    u = jax.vmap(jnp.where, axis_name='i')(u < nx0s, QLs, QRs)
    return u[...,None,None]

#@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))
def init_multi_HD_KH(u, xc, yc, zc, numbers=10000, init_key=2022, M0=0.1, dkMx=2., kmax=4., gamma=1.666666667):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    assert zc.shape[0] == 1, 'nz is assumed to be 1!!'
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    def __create_KH_init(u, dk, kk):
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        d0_u = 2./(dk + 1.)
        d0_d = dk * d0_u
        d0 = 0.5 * (d0_u + d0_d)
        ux = 1.
        cs = ux/M0
        p0 = cs**2 * d0/gamma
        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        qLx = dx * nx
        qLy = dy * ny
        kx = kk * 2. * jnp.pi / qLx
        dl = 5.e-3 * qLy
        # (numbers, nx)
        bound = 0.5 * qLy + dl * jnp.sin(kx * xc)  # assuming yL = 0

        vx = jnp.zeros([nx, ny, nz])
        dd = jnp.zeros([nx, ny, nz])
        for i in range(nx):
            _vx = jnp.where(yc > bound[i], ux, -ux)
            _dd = jnp.where(yc > bound[i], d0_u, d0_d)
            vx = vx.at[i, :, :].set(_vx[:, None])
            dd = dd.at[i, :, :].set(_dd[:, None])

        u = u.at[0, 2:-2, 2:-2, 2:-2].set(dd)
        u = u.at[1, 2:-2, 2:-2, 2:-2].set(vx)
        u = u.at[2].set(0.)
        u = u.at[3].add(0.)
        u = u.at[4].add(p0)
        return u

    # create random density ratio
    key = random.PRNGKey(init_key)
    dk = random.uniform(key, shape=([numbers, 1]), minval=1. / dkMx, maxval=dkMx)
    #create random wave-numbers
    key, subkey = random.split(key)
    kk = random.randint(key, shape=([numbers, 1]), minval=1, maxval=kmax)
    print('vmap...')
    u = jax.vmap(__create_KH_init, axis_name='i')(u, dk, kk)

    return u

#@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def init_multi_HD_2DTurb(u, xc, yc, zc, numbers=10000, init_key=2022, M0=0.1, k_tot=4., gamma=1.666666667):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    assert zc.shape[0] == 1, 'nz is assumed to be 1!!'
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    def __create_2DTurb_init(u, keys):
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        d0 = 1.
        cs = 1./M0
        u0 = 1.  # fixed
        p0 = cs ** 2 * d0 / gamma

        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]

        qLx = dx * nx
        qLy = dy * ny

        ## random velocity field
        vx, vy = jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz])

        key = random.PRNGKey(keys)

        kx0 = jnp.pi * 2. / qLx
        ky0 = jnp.pi * 2. / qLy

        for j in range(-k_tot, k_tot + 1):
            ky = ky0 * j  # from 1 to k_tot
            for i in range(-k_tot, k_tot + 1):
                kx = kx0 * i  # from 1 to k_tot
                if i * j == 0:  # avoiding uniform velocity
                    continue
                # random phase
                key, subkey = random.split(key)
                phs = 2. * jnp.pi * random.uniform(key, shape=[2])  # (vi, k)

                uk = 1. / jnp.sqrt(jnp.sqrt(kx ** 2 + ky ** 2))
                kdx = kx * xc[:, None, None] + ky * yc[None, :, None]
                vx += uk * jnp.sin(kdx + phs[0])
                vy += uk * jnp.sin(kdx + phs[1])

        del (kdx, uk, phs)

        # Helmholtz decomposition to subtract expansion: k.vk
        dfx, dfy = 1. / qLx, 1. / qLy
        fx = dfx * (jnp.arange(nx) - 1. - nx // 2)
        fy = dfy * (jnp.arange(ny) - 1. - ny // 2)

        vkx = jnp.fft.fftn(vx) * dx * dy
        vky = jnp.fft.fftn(vy) * dx * dy

        # shift to kxi=0 is at the center
        vkx = jnp.fft.fftshift(vkx)
        vky = jnp.fft.fftshift(vky)

        fi = fx[:, None, None] ** 2 + fy[None, :, None] ** 2
        fi = jnp.where(fi > 1.e-8, 1. / fi, 0.)

        fdv = (fx[:, None, None] * vkx + fy[None, :, None] * vky) * fi
        vkx -= fdv * fx[:, None, None]
        vky -= fdv * fy[None, :, None]
        del (fi, fdv)

        # shift back to original order
        vkx = jnp.fft.ifftshift(vkx)
        vky = jnp.fft.ifftshift(vky)

        # inverse FFT
        vx = jnp.fft.ifftn(vkx).real * dfx * dfy
        vy = jnp.fft.ifftn(vky).real * dfx * dfy

        # renormalize total velocity
        vtot = jnp.sqrt(vx ** 2 + vy ** 2).mean()
        vx *= u0 / vtot
        vy *= u0 / vtot

        u = u.at[0].set(d0)
        u = u.at[1, 2:-2, 2:-2, 2:-2].set(vx)
        u = u.at[2, 2:-2, 2:-2, 2:-2].set(vy)
        u = u.at[4].add(p0)
        return u

    key = random.PRNGKey(init_key)
    keys = random.randint(key, [numbers,], minval=0, maxval=10000000)
    u = jax.vmap(__create_2DTurb_init, axis_name='i')(u, keys)

    return u

def init_multi_HD_2DRand(u, xc, yc, zc, numbers=10000, init_key=2022, M0=0.1, k_tot=4., gamma=1.666666667,
                         dMx=1.e1, TMx=1.e1):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    assert zc.shape[0] == 1, 'nz is assumed to be 1!!'
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    def _pass(carry):
        return carry

    def select_W(carry):
        def _window(carry):
            xx, yy, val, xL, xR, yL, yR, trns = carry
            x_win = 0.5 * (jnp.tanh((xx - xL) / trns) - jnp.tanh((xx - xR) / trns))
            y_win = 0.5 * (jnp.tanh((yy - yL) / trns) - jnp.tanh((yy - yR) / trns))
            val = x_win[:, None] * y_win[None, :]
            return xx, yy, val, xL, xR, yL, yR, trns

        cond, value, xx, yy, xL, xR, yL, yR, trns = carry

        carry = xx, yy, value, xL, xR, yL, yR, trns
        xx, yy, value, xL, xR, yL, yR, trns = lax.cond(cond == 1, _window, _pass, carry)
        return cond, value, xx, yy, xL, xR, yL, yR, trns

    def __create_2DRand_init(u, d0, T0, delD, delP, keys):
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]

        p0 = d0 * T0
        cs = jnp.sqrt(T0 * gamma)
        u0 = M0 * cs

        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]

        qLx = dx * nx
        qLy = dy * ny

        ## random velocity field
        d, p, vx, vy = jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz])

        key = random.PRNGKey(keys)
        kx0 = jnp.pi * 2. / qLx
        ky0 = jnp.pi * 2. / qLy

        for j in range(-k_tot, k_tot + 1):
            ky = ky0 * j  # from 1 to k_tot
            for i in range(-k_tot, k_tot + 1):
                kx = kx0 * i  # from 1 to k_tot
                if i * j == 0:  # avoiding uniform velocity
                    continue
                # random phase
                key, subkey = random.split(key)
                phs = 2. * jnp.pi * random.uniform(key, shape=[4])  # (vi, k)

                uk = 1. / jnp.sqrt(jnp.sqrt(kx ** 2 + ky ** 2))
                kdx = kx * xc[:, None, None] + ky * yc[None, :, None]
                vx += uk * jnp.sin(kdx + phs[0])
                vy += uk * jnp.sin(kdx + phs[1])
                p += uk * jnp.sin(kdx + phs[2])
                d += uk * jnp.sin(kdx + phs[3])

        del (kdx, uk, phs)

        # renormalize total velocity
        vtot = jnp.sqrt(vx ** 2 + vy ** 2).mean()
        vx *= u0 / vtot
        vy *= u0 / vtot
        #d = d0 + delD * d / jnp.abs(d).mean()
        #p = p0 + delP * p / jnp.abs(p).mean()
        d = d0 * (1. + delD * d / jnp.abs(d).mean())
        p = p0 * (1. + delP * p / jnp.abs(p).mean())

        u = u.at[0, 2:-2, 2:-2, 2:-2].set(d)
        u = u.at[1, 2:-2, 2:-2, 2:-2].set(vx)
        u = u.at[2, 2:-2, 2:-2, 2:-2].set(vy)
        u = u.at[4, 2:-2, 2:-2, 2:-2].set(p)
        return u

    key = random.PRNGKey(init_key)
    d0 = random.uniform(key, shape=([numbers, 1]), minval=1.e-1, maxval=dMx)
    key, subkey = random.split(key)
    delD = random.uniform(key, shape=([numbers, 1]), minval=1.e-2, maxval=0.2)
    key, subkey = random.split(key)
    T0 = random.uniform(key, shape=([numbers, 1]), minval=1.e-1, maxval=TMx)
    key, subkey = random.split(key)
    delP = random.uniform(key, shape=([numbers, 1]), minval=1.e-2, maxval=0.2)

    key, subkey = random.split(key)
    keys = random.randint(key, shape=([numbers, ]), minval=0, maxval=10000000)
    u = jax.vmap(__create_2DRand_init, axis_name='i')(u, d0, T0, delD, delP, keys)


    # perform window function
    key, subkey = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.5, 0.5]), shape=([numbers]))
    mask = jnp.ones([numbers, xc.shape[0], yc.shape[0]])
    xL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    xR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    key, subkey = random.split(key)
    yL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    yR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)  # add batch
    _yc = jnp.repeat(yc[None, :], numbers, axis=0)  # add batch
    trns = 0.01 * jnp.ones_like(cond)
    carry = cond, mask, _xc, _yc, xL, xR, yL, yR, trns
    cond, mask, _xc, _yc, xL, xR, yL, yR, trns = vmap(select_W, 0, 0)(carry)

    u = u.at[:, :, 2:-2, 2:-2, 2:-2].set(u[:, :, 2:-2, 2:-2, 2:-2] * mask[:, None, :, :, None])
    u = u.at[:, 0, 2:-2, 2:-2, 2:-2].add(d0[:, :, None, None] * (1. - mask[:, :, :, None]))
    u = u.at[:, 4, 2:-2, 2:-2, 2:-2].add(d0[:, :, None, None] * T0[:, :, None, None] * (1. - mask[:, :, :, None]))

    return u

def init_multi_HD_3DTurb(u, xc, yc, zc, numbers=100, init_key=2022, M0=0.1, k_tot=4., gamma=1.666666667):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    def __create_3DTurb_init(u, keys):
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]
        d0 = 1.
        cs = 1./M0
        u0 = 1.  # fixed
        p0 = cs ** 2 * d0 / gamma

        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        dz = zc[1] - zc[0]

        qLx = dx * nx
        qLy = dy * ny
        qLz = dz * nz

        ## random velocity field
        vx, vy, vz = jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz])

        key = random.PRNGKey(keys)

        kx0 = jnp.pi * 2. / qLx
        ky0 = jnp.pi * 2. / qLy
        kz0 = jnp.pi * 2. / qLz

        for k in range(-k_tot, k_tot + 1):
            kz = kz0 * k  # from 1 to k_tot
            for j in range(-k_tot, k_tot + 1):
                ky = ky0 * j  # from 1 to k_tot
                for i in range(-k_tot, k_tot + 1):
                    kx = kx0 * i  # from 1 to k_tot
                    if i * j * k == 0:  # avoiding uniform velocity
                        continue
                    # random phase
                    key, subkey = random.split(key)
                    phs = 2. * jnp.pi * random.uniform(key, shape=[3])  # (vi, k)

                    uk = 1./jnp.sqrt(kx**2 + ky**2 + kz**2)
                    kdx = kx * xc[:,None,None] + ky * yc[None,:,None] + kz * zc[None,None,:]
                    vx += uk * jnp.sin(kdx + phs[0])
                    vy += uk * jnp.sin(kdx + phs[1])
                    vz += uk * jnp.sin(kdx + phs[2])

        del(kdx, uk, phs)

        # Helmholtz decomposition to subtract expansion: k.vk
        dfx, dfy, dfz = 1./qLx, 1./qLy, 1./qLz
        fx = dfx * (jnp.arange(nx) - 1. - nx//2)
        fy = dfy * (jnp.arange(ny) - 1. - ny//2)
        fz = dfz * (jnp.arange(nz) - 1. - nz//2)

        vkx = jnp.fft.fftn(vx) * dx * dy * dz
        vky = jnp.fft.fftn(vy) * dx * dy * dz
        vkz = jnp.fft.fftn(vz) * dx * dy * dz

        # shift to kxi=0 is at the center
        vkx = jnp.fft.fftshift(vkx)
        vky = jnp.fft.fftshift(vky)
        vkz = jnp.fft.fftshift(vkz)

        fi = fx[:,None,None]**2 + fy[None,:,None]**2 + fz[None,None,:]**2
        fi = jnp.where(fi > 1.e-8, 1./fi, 0.)

        fdv = (fx[:,None,None] * vkx + fy[None,:,None] * vky + fz[None,None,:] * vkz) * fi
        vkx -= fdv * fx[:,None,None]
        vky -= fdv * fy[None,:,None]
        vkz -= fdv * fz[None,None,:]
        del(fi, fdv)

        # shift back to original order
        vkx = jnp.fft.ifftshift(vkx)
        vky = jnp.fft.ifftshift(vky)
        vkz = jnp.fft.ifftshift(vkz)

        # inverse FFT
        vx = jnp.fft.ifftn(vkx).real * dfx * dfy * dfz
        vy = jnp.fft.ifftn(vky).real * dfx * dfy * dfz
        vz = jnp.fft.ifftn(vkz).real * dfx * dfy * dfz

        # renormalize total velocity
        vtot = jnp.sqrt(vx**2 + vy**2 + vz**2).mean()
        vx *= u0 / vtot
        vy *= u0 / vtot
        vz *= u0 / vtot

        u = u.at[0].set(d0)
        u = u.at[1, 2:-2, 2:-2, 2:-2].set(vx)
        u = u.at[2, 2:-2, 2:-2, 2:-2].set(vy)
        u = u.at[3, 2:-2, 2:-2, 2:-2].set(vz)
        u = u.at[4].add(p0)
        return u

    key = random.PRNGKey(init_key)
    keys = random.randint(key, [numbers,], minval=0, maxval=10000000)
    u = jax.vmap(__create_3DTurb_init, axis_name='i')(u, keys)

    return u

def init_multi_HD_3DRand(u, xc, yc, zc, numbers=10000, init_key=2022, M0=0.1, k_tot=4., gamma=1.666666667,
                         dMx=1.e1, TMx=1.e1):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    assert numbers % jax.device_count() == 0, 'numbers should be : GPUs x integer!!'

    def _pass(carry):
        return carry

    def select_W(carry):
        def _window(carry):
            xx, yy, zz, val, xL, xR, yL, yR, zL, zR, trns = carry
            x_win = 0.5 * (jnp.tanh((xx - xL) / trns) - jnp.tanh((xx - xR) / trns))
            y_win = 0.5 * (jnp.tanh((yy - yL) / trns) - jnp.tanh((yy - yR) / trns))
            z_win = 0.5 * (jnp.tanh((zz - zL) / trns) - jnp.tanh((zz - zR) / trns))
            val = x_win[:, None, None] * y_win[None, :, None] * z_win[None, None, :]
            return xx, yy, zz, val, xL, xR, yL, yR, zL, zR, trns

        cond, value, xx, yy, zz, xL, xR, yL, yR, zL, zR, trns = carry

        carry = xx, yy, zz, value, xL, xR, yL, yR, zL, zR, trns
        xx, yy, zz, value, xL, xR, yL, yR, zL, zR, trns = lax.cond(cond == 1, _window, _pass, carry)
        return cond, value, xx, yy, zz, xL, xR, yL, yR, zL, zR, trns

    def __create_3DRand_init(u, d0, T0, delD, delP, keys):
        nx, ny, nz = xc.shape[0], yc.shape[0], zc.shape[0]

        p0 = d0 * T0
        cs = jnp.sqrt(T0 * gamma)
        u0 = M0 * cs

        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        dz = zc[1] - zc[0]

        qLx = dx * nx
        qLy = dy * ny
        qLz = dz * nz

        ## random velocity field
        d, p, vx, vy, vz = jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz]), \
                           jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz]), jnp.zeros([nx, ny, nz])

        key = random.PRNGKey(keys)
        kx0 = jnp.pi * 2. / qLx
        ky0 = jnp.pi * 2. / qLy
        kz0 = jnp.pi * 2. / qLz

        for k in range(-k_tot, k_tot + 1):
            kz = kz0 * k  # from 1 to k_tot
            for j in range(-k_tot, k_tot + 1):
                ky = ky0 * j  # from 1 to k_tot
                for i in range(-k_tot, k_tot + 1):
                    kx = kx0 * i  # from 1 to k_tot
                    if i * j * k == 0:  # avoiding uniform velocity
                        continue
                    # random phase
                    key, subkey = random.split(key)
                    phs = 2. * jnp.pi * random.uniform(key, shape=[5])  # (vi, k)

                    uk = 1./jnp.sqrt(kx**2 + ky**2 + kz**2)
                    kdx = kx * xc[:,None,None] + ky * yc[None,:,None] + kz * zc[None,None,:]
                    vx += uk * jnp.sin(kdx + phs[0])
                    vy += uk * jnp.sin(kdx + phs[1])
                    vz += uk * jnp.sin(kdx + phs[2])
                    p += uk * jnp.sin(kdx + phs[3])
                    d += uk * jnp.sin(kdx + phs[4])

        del(kdx, uk, phs)

        # renormalize total velocity
        vtot = jnp.sqrt(vx ** 2 + vy ** 2 + vz ** 2).mean()
        vx *= u0 / vtot
        vy *= u0 / vtot
        vz *= u0 / vtot
        #d = d0 + delD * d / jnp.abs(d).mean()
        #p = p0 + delP * p / jnp.abs(p).mean()
        d = d0 * (1. + delD * d / jnp.abs(d).mean())
        p = p0 * (1. + delP * p / jnp.abs(p).mean())

        u = u.at[0, 2:-2, 2:-2, 2:-2].set(d)
        u = u.at[1, 2:-2, 2:-2, 2:-2].set(vx)
        u = u.at[2, 2:-2, 2:-2, 2:-2].set(vy)
        u = u.at[3, 2:-2, 2:-2, 2:-2].set(vz)
        u = u.at[4, 2:-2, 2:-2, 2:-2].set(p)
        return u

    key = random.PRNGKey(init_key)
    d0 = random.uniform(key, shape=([numbers, 1]), minval=1.e-1, maxval=dMx)
    key, subkey = random.split(key)
    delD = random.uniform(key, shape=([numbers, 1]), minval=1.e-2, maxval=0.2)
    key, subkey = random.split(key)
    T0 = random.uniform(key, shape=([numbers, 1]), minval=1.e-1, maxval=TMx)
    key, subkey = random.split(key)
    delP = random.uniform(key, shape=([numbers, 1]), minval=1.e-2, maxval=0.2)

    key, subkey = random.split(key)
    keys = random.randint(key, [numbers,], minval=0, maxval=10000000)
    u = jax.vmap(__create_3DRand_init, axis_name='i')(u, d0, T0, delD, delP, keys)

    # perform window function
    key, subkey = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.5, 0.5]), shape=([numbers]))
    mask = jnp.ones([numbers, xc.shape[0], yc.shape[0], zc.shape[0]])
    xL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    xR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    key, subkey = random.split(key)
    yL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    yR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    key, subkey = random.split(key)
    zL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    zR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)
    _yc = jnp.repeat(yc[None, :], numbers, axis=0)
    _zc = jnp.repeat(zc[None, :], numbers, axis=0)
    trns = 0.01 * jnp.ones_like(cond)
    carry = cond, mask, _xc, _yc, _zc, xL, xR, yL, yR, zL, zR, trns
    cond, mask, _xc, _yc, _zc, xL, xR, yL, yR, zL, zR, trns = vmap(select_W, 0, 0)(carry)

    u = u.at[:, :, 2:-2, 2:-2, 2:-2].set(u[:, :, 2:-2, 2:-2, 2:-2] * mask[:, None, :, :, :])
    u = u.at[:, 0, 2:-2, 2:-2, 2:-2].add(d0[:, :, None, None] * (1. - mask[:, :, :, :]))
    u = u.at[:, 4, 2:-2, 2:-2, 2:-2].add(d0[:, :, None, None] * T0[:, :, None, None] * (1. - mask[:, :, :, :]))

    return u

def bc(u, dx, Ncell, mode='periodic'):
    _u = jnp.zeros(Ncell+4) # because of 2nd-order precision in space
    _u = _u.at[2:Ncell+2].set(u)
    if mode=='periodic':  # periodic boundary condition
        _u = _u.at[0:2].set(u[-2:])  # left hand side
        _u = _u.at[Ncell + 2:Ncell + 4].set(u[0:2])  # right hand side
    elif mode=='reflection':
        _u = _u.at[0].set(- u[3])  # left hand side
        _u = _u.at[1].set(- u[2])  # left hand side
        _u = _u.at[-2].set(- u[-3])  # right hand side
        _u = _u.at[-1].set(- u[-4])  # right hand side
    elif mode=='copy':
        _u = _u.at[0].set(u[3])  # left hand side
        _u = _u.at[1].set(u[2])  # left hand side
        _u = _u.at[-2].set(u[-3])  # right hand side
        _u = _u.at[-1].set(u[-4])  # right hand side

    return _u

def bc_2D(_u, mode='trans'):
    Nx, Ny = _u.shape
    u = jnp.zeros([Nx + 4, Ny + 4]) # because of 2nd-order precision in space
    u = u.at[2:-2, 2:-2].set(_u)
    Nx += 2
    Ny += 2

    if mode=='periodic': # periodic boundary condition
        # left hand side
        u = u.at[0:2, 2:-2].set(u[Nx-2:Nx, 2:-2])  # x
        u = u.at[2:-2, 0:2].set(u[2:-2, Ny-2:Ny])  # y
        # right hand side
        u = u.at[Nx:Nx+2, 2:-2].set(u[2:4, 2:-2])
        u = u.at[2:-2, Ny:Ny+2].set(u[2:-2, 2:4])
    elif mode=='trans': # periodic boundary condition
        # left hand side
        u = u.at[0, 2:-2].set(u[3, 2:-2])  # x
        u = u.at[2:-2, 0].set(u[2:-2, 3])  # y
        u = u.at[1, 2:-2].set(u[2, 2:-2])  # x
        u = u.at[2:-2, 1].set(u[2:-2, 2])  # y
        # right hand side
        u = u.at[-2, 2:-2].set(u[-3, 2:-2])
        u = u.at[2:-2, -2].set(u[2:-2, -3])
        u = u.at[-1, 2:-2].set(u[-4, 2:-2])
        u = u.at[2:-2, -1].set(u[2:-2, -4])
    elif mode=='Neumann': # periodic boundary condition
        # left hand side
        u = u.at[0, 2:-2].set(0.)  # x
        u = u.at[2:-2, 0].set(0.)  # y
        u = u.at[1, 2:-2].set(0.)  # x
        u = u.at[2:-2, 1].set(0.)  # y
        # right hand side
        u = u.at[-2, 2:-2].set(0.)
        u = u.at[2:-2, -2].set(0.)
        u = u.at[-1, 2:-2].set(0.)
        u = u.at[2:-2, -1].set(0.)
    return u

def bc_HD(u, mode):
    _, Nx, Ny, Nz = u.shape
    Nx -= 2
    Ny -= 2
    Nz -= 2
    if mode=='periodic': # periodic boundary condition
        # left hand side
        u = u.at[:, 0:2, 2:-2, 2:-2].set(u[:, Nx-2:Nx, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 0:2, 2:-2].set(u[:, 2:-2, Ny-2:Ny, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 0:2].set(u[:, 2:-2, 2:-2, Nz-2:Nz])  # z
        # right hand side
        u = u.at[:, Nx:Nx+2, 2:-2, 2:-2].set(u[:, 2:4, 2:-2, 2:-2])
        u = u.at[:, 2:-2, Ny:Ny+2, 2:-2].set(u[:, 2:-2, 2:4, 2:-2])
        u = u.at[:, 2:-2, 2:-2, Nz:Nz+2].set(u[:, 2:-2, 2:-2, 2:4])
    elif mode=='trans': # periodic boundary condition
        # left hand side
        u = u.at[:, 0, 2:-2, 2:-2].set(u[:, 3, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 0, 2:-2].set(u[:, 2:-2, 3, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 0].set(u[:, 2:-2, 2:-2, 3])  # z
        u = u.at[:, 1, 2:-2, 2:-2].set(u[:, 2, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 1, 2:-2].set(u[:, 2:-2, 2, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 1].set(u[:, 2:-2, 2:-2, 2])  # z
        # right hand side
        u = u.at[:, -2, 2:-2, 2:-2].set(u[:, -3, 2:-2, 2:-2])
        u = u.at[:, 2:-2, -2, 2:-2].set(u[:, 2:-2, -3, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -2].set(u[:, 2:-2, 2:-2, -3])
        u = u.at[:, -1, 2:-2, 2:-2].set(u[:, -4, 2:-2, 2:-2])
        u = u.at[:, 2:-2, -1, 2:-2].set(u[:, 2:-2, -4, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -1].set(u[:, 2:-2, 2:-2, -4])
    elif mode=='KHI': # x: periodic, y, z : trans
        # left hand side
        u = u.at[:, 0:2, 2:-2, 2:-2].set(u[:, Nx - 2:Nx, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 0, 2:-2].set(u[:, 2:-2, 3, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 0].set(u[:, 2:-2, 2:-2, 3])  # z
        u = u.at[:, 2:-2, 1, 2:-2].set(u[:, 2:-2, 2, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 1].set(u[:, 2:-2, 2:-2, 2])  # z
        # right hand side
        u = u.at[:, Nx:Nx + 2, 2:-2, 2:-2].set(u[:, 2:4, 2:-2, 2:-2])
        u = u.at[:, 2:-2, -2, 2:-2].set(u[:, 2:-2, -3, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -2].set(u[:, 2:-2, 2:-2, -3])
        u = u.at[:, 2:-2, -1, 2:-2].set(u[:, 2:-2, -4, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -1].set(u[:, 2:-2, 2:-2, -4])
    return u

def bc_HD_vis(u, if_periodic=True):  # for viscosity
    """
    for the moment, assuming periodic/copy boundary
    seemingly, copy boundary does not work well...
    """
    _, Nx, Ny, Nz = u.shape
    Nx -= 2
    Ny -= 2
    Nz -= 2

    if if_periodic:
        u = u.at[:, 0:2, 0:2, 2:-2].set(u[:, Nx - 2:Nx, Ny - 2:Ny, 2:-2])  # xByB
        u = u.at[:, 0:2, 2:-2, 0:2].set(u[:, Nx - 2:Nx, 2:-2, Nz - 2:Nz])  # xBzB
        u = u.at[:, 0:2, Ny:Ny + 2, 2:-2].set(u[:, Nx - 2:Nx, 2:4, 2:-2])  # xByT
        u = u.at[:, 0:2, 2:-2, Nz:Nz + 2].set(u[:, Nx - 2:Nx, 2:-2, 2:4])  # xBzT
        u = u.at[:, Nx:Nx + 2, 0:2, 2:-2].set(u[:, 2:4, Ny - 2:Ny, 2:-2])  # xTyB
        u = u.at[:, Nx:Nx + 2, 2:-2, 0:2].set(u[:, 2:4, 2:-2, Nz - 2:Nz])  # xTzB
        u = u.at[:, Nx:Nx + 2, Ny:Ny + 2, 2:-2].set(u[:, 2:4, 2:4, 2:-2])  # xTyT
        u = u.at[:, Nx:Nx + 2, 2:-2, Nz:Nz + 2].set(u[:, 2:4, 2:-2, 2:4])  # xTzT
    else: # trans
        u = u.at[:, 0:2, 0:2, 2:-2].set(u[:, 4:2, 4:2, 2:-2])  # xByT
        u = u.at[:, 0:2, 2:-2, 0:2].set(u[:, 4:2, 2:-2, 4:2])  # xBzB
        u = u.at[:, 0:2, Ny:Ny + 2, 2:-2].set(u[:, 4:2, Ny:Ny-2, 2:-2])  # xByB
        u = u.at[:, 0:2, 2:-2, Nz:Nz + 2].set(u[:, 4:2, 2:-2, Nz:Nz-2])  # xBzT
        u = u.at[:, Nx:Nx + 2, 0:2, 2:-2].set(u[:, Nx:Nx-2, 4:2, 2:-2])  # xTyB
        u = u.at[:, Nx:Nx + 2, 2:-2, 0:2].set(u[:, Nx:Nx-2, 2:-2, 4:2])  # xTzB
        u = u.at[:, Nx:Nx + 2, Ny:Ny + 2, 2:-2].set(u[:, Nx:Nx-2, Ny:Ny-2, 2:-2])  # xTyT
        u = u.at[:, Nx:Nx + 2, 2:-2, Nz:Nz + 2].set(u[:, Nx:Nx-2, 2:-2, Nz:Nz-2])  # xTzT

    return u

def VLlimiter(a, b, c, alpha=2.):
    return jnp.sign(c)\
           *(0.5 + 0.5*jnp.sign(a*b))\
           *jnp.minimum(alpha*jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))

def limiting(u, Ncell, if_second_order):
    # under construction
    duL = u[1:Ncell + 3] - u[0:Ncell + 2]
    duR = u[2:Ncell + 4] - u[1:Ncell + 3]
    duM = (u[2:Ncell + 4] - u[0:Ncell + 2])*0.5
    gradu = VLlimiter(duL, duR, duM) * if_second_order
    # -1:Ncell
    #uL, uR = jnp.zeros(Ncell+4), jnp.zeros(Ncell+4)
    uL, uR = jnp.zeros_like(u), jnp.zeros_like(u)
    uL = uL.at[1:Ncell+3].set(u[1:Ncell+3] - 0.5*gradu)  # left of cell
    uR = uR.at[1:Ncell+3].set(u[1:Ncell+3] + 0.5*gradu)  # right of cell
    return uL, uR

def limiting_HD(u, if_second_order):
    nd, nx, ny, nz = u.shape
    uL, uR = u, u
    nx -= 4

    duL = u[:, 1:nx + 3, :, :] - u[:, 0:nx + 2, :, :]
    duR = u[:, 2:nx + 4, :, :] - u[:, 1:nx + 3, :, :]
    duM = (u[:, 2:nx + 4, :, :] - u[:, 0:nx + 2, :, :]) * 0.5
    gradu = VLlimiter(duL, duR, duM) * if_second_order
    # -1:Ncell
    uL = uL.at[:, 1:nx + 3, :, :].set(u[:, 1:nx + 3, :, :] - 0.5*gradu)  # left of cell
    uR = uR.at[:, 1:nx + 3, :, :].set(u[:, 1:nx + 3, :, :] + 0.5*gradu)  # right of cell

    uL = jnp.where(uL[0] > 0., uL, u)
    uL = jnp.where(uL[4] > 0., uL, u)
    uR = jnp.where(uR[0] > 0., uR, u)
    uR = jnp.where(uR[4] > 0., uR, u)

    return uL, uR

def save_data(u, xc, i_save, save_dir, dt_save=None, if_final=False):
    if if_final:
        jnp.save(save_dir+'/x_coordinate', xc)
        #
        tc = jnp.arange(i_save+1)*dt_save
        jnp.save(save_dir+'/t_coordinate', tc)
        #
        flnm = save_dir+'/Data_'+str(i_save).zfill(4)
        jnp.save(flnm, u)
    else:
        flnm = save_dir+'/Data_'+str(i_save).zfill(4)
        jnp.save(flnm, u)

def save_data_HD(u, xc, yc, zc, i_save, save_dir, dt_save=None, if_final=False):
    if if_final:
        jnp.save(save_dir+'/x_coordinate', xc)
        jnp.save(save_dir+'/y_coordinate', yc)
        jnp.save(save_dir+'/z_coordinate', zc)
        #
        tc = jnp.arange(i_save+1)*dt_save
        jnp.save(save_dir+'/t_coordinate', tc)
        #
        flnm = save_dir+'/Data_'+str(i_save).zfill(4)
        jnp.save(flnm, u)
    else:
        flnm = save_dir+'/Data_'+str(i_save).zfill(4)
        jnp.save(flnm, u)

def Courant(u, dx):
    stability_adv = dx/(jnp.max(jnp.abs(u)) + 1.e-8)
    return stability_adv

def Courant_diff(dx, epsilon=1.e-3):
    stability_dif = 0.5*dx**2/(epsilon + 1.e-8)
    return stability_dif

def Courant_diff_2D(dx, dy, epsilon=1.e-3):
    stability_dif_x = 0.5*dx**2/(epsilon + 1.e-8)
    stability_dif_y = 0.5*dy**2/(epsilon + 1.e-8)
    return jnp.min(jnp.array([stability_dif_x, stability_dif_y]))

def Courant_HD(u, dx, dy, dz, gamma):
    cs = jnp.sqrt(gamma*u[4]/u[0])  # sound velocity
    stability_adv_x = dx/(jnp.max(cs + jnp.abs(u[1])) + 1.e-8)
    stability_adv_y = dy/(jnp.max(cs + jnp.abs(u[2])) + 1.e-8)
    stability_adv_z = dz/(jnp.max(cs + jnp.abs(u[3])) + 1.e-8)
    stability_adv = jnp.min(jnp.array([stability_adv_x, stability_adv_y, stability_adv_z]))
    return stability_adv

def Courant_vis_HD(dx, dy, dz, eta, zeta):
    #visc = jnp.max(jnp.array([eta, zeta]))
    visc = 4. / 3. * eta + zeta  # maximum
    stability_dif_x = 0.5*dx**2/(visc + 1.e-8)
    stability_dif_y = 0.5*dy**2/(visc + 1.e-8)
    stability_dif_z = 0.5*dz**2/(visc + 1.e-8)
    stability_dif = jnp.min(jnp.array([stability_dif_x, stability_dif_y, stability_dif_z]))
    return stability_dif

