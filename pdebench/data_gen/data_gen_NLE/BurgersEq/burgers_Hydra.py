#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     burgers_Hydra.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import time
import sys
from math import ceil

# Hydra
from omegaconf import DictConfig, OmegaConf
import hydra

import jax
import jax.numpy as jnp
from jax import device_put, lax

sys.path.append('..')
from utils import init, Courant, Courant_diff, save_data, bc, limiting

def _pass(carry):
    return carry

# Init arguments with Hydra
@hydra.main(config_path="config")
def main(cfg: DictConfig) -> None:
    # basic parameters
    pi_inv = 1. / jnp.pi
    dx = (cfg.args.xR - cfg.args.xL) / cfg.args.nx
    dx_inv = 1. / dx

    # cell edge coordinate
    xe = jnp.linspace(cfg.args.xL, cfg.args.xR, cfg.args.nx + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx

    show_steps = cfg.args.show_steps
    ini_time = cfg.args.ini_time
    fin_time = cfg.args.fin_time
    dt_save = cfg.args.dt_save
    CFL = cfg.args.CFL

    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    @jax.jit
    def evolve(u):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.
        uu = jnp.zeros([it_tot, u.shape[0]])
        uu = uu.at[0].set(u)

        tm_ini = time.time()

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _save(_carry):
                u, tsave, i_save, uu = _carry
                uu = uu.at[i_save].set(u)
                tsave += dt_save
                i_save += 1
                return (u, tsave, i_save, uu)

            t, tsave, steps, i_save, dt, u, uu = carry

            # if save data
            carry = (u, tsave, i_save, uu)
            u, tsave, i_save, uu = lax.cond(t >= tsave, _save, _pass, carry)

            carry = (u, t, dt, steps, tsave)
            u, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, u, uu)

        carry = t, tsave, steps, i_save, dt, u, uu
        t, tsave, steps, i_save, dt, u, uu = lax.while_loop(cond_fun, _body_fun, carry)
        uu = uu.at[-1].set(u)

        tm_fin = time.time()
        print('total elapsed time is {} sec'.format(tm_fin - tm_ini))
        return uu, t

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave = carry
        dt_adv = Courant(u, dx) * CFL
        dt_dif = Courant_diff(dx, cfg.args.epsilon*pi_inv) * CFL
        dt = jnp.min(jnp.array([dt_adv, dt_dif, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt = carry
            # preditor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5)
            # update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt)
            return u, dt

        carry = u, dt
        u, dt = lax.cond(dt > 1.e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave

    @jax.jit
    def update(u, u_tmp, dt):
        f = flux(u_tmp)
        u -= dt * dx_inv * (f[1:cfg.args.nx + 1] - f[0:cfg.args.nx])
        return u

    def flux(u):
        _u = bc(u, dx, Ncell=cfg.args.nx) # index 2 for _U is equivalent with index 0 for u
        uL, uR = limiting(_u, cfg.args.nx, if_second_order=1.)
        fL = 0.5*uL**2
        fR = 0.5*uR**2
        # upwind advection scheme
        f_upwd = (fR[1:cfg.args.nx+2] + fL[2:cfg.args.nx+3]
               - 0.5*jnp.abs(uL[2:cfg.args.nx+3] + uR[1:cfg.args.nx+2])*(uL[2:cfg.args.nx+3] - uR[1:cfg.args.nx+2]))
        # source term
        f_upwd += - cfg.args.epsilon*pi_inv*(_u[2:cfg.args.nx+3] - _u[1:cfg.args.nx+2])*dx_inv
        return f_upwd

    u = init(xc=xc, mode=cfg.args.init_mode, u0=cfg.args.u0, du=cfg.args.du)
    u = device_put(u)  # putting variables in GPU (not necessary??)
    uu, t = evolve(u)
    print('final time is: {0:.3f}'.format(t))

    print('data saving...')
    cwd = hydra.utils.get_original_cwd() + '/'
    if cfg.args.init_mode=='sinsin':
        jnp.save(cwd + cfg.args.save + '/Burgers_' + cfg.args.init_mode + '_u' + str(cfg.args.u0) + '_du' + str(
            cfg.args.du) + '_Nu' + str(cfg.args.epsilon), uu)
    else:
        jnp.save(cwd + cfg.args.save + '/Burgers_' + cfg.args.init_mode + '_u' + str(cfg.args.u0) + '_Nu' + str(
            cfg.args.epsilon), uu)
    jnp.save(cwd + cfg.args.save+'/x_coordinate', xc)
    jnp.save(cwd + cfg.args.save+'/t_coordinate', tc)

if __name__=='__main__':
    main()
