import deepxde as dde
import numpy as np
import torch


def reaction_1(u1, u2):
    k = 5e-3

    return u1 - (u1 * u1 * u1) - k - u2


def reaction_2(u1, u2):
    return u1 - u2


def pde_diffusion_reaction(x, y):

    d1 = 1e-3
    d2 = 5e-3

    du1_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)
    du1_yy = dde.grad.hessian(y, x, i=1, j=1, component=0)
    du2_xx = dde.grad.hessian(y, x, i=0, j=0, component=1)
    du2_yy = dde.grad.hessian(y, x, i=1, j=1, component=1)

    # TODO: check indices of jacobian
    du1_t = dde.grad.jacobian(y, x, i=0, j=2)
    du2_t = dde.grad.jacobian(y, x, i=1, j=2)

    u1 = y[..., 0].unsqueeze(1)
    u2 = y[..., 1].unsqueeze(1)

    eq1 = du1_t - reaction_1(u1, u2) - d1 * (du1_xx + du1_yy)
    eq2 = du2_t - reaction_2(u1, u2) - d2 * (du2_xx + du2_yy)

    return eq1 + eq2


def pde_diffusion_sorption(x, y):
    D: float = 5e-4
    por: float = 0.29
    rho_s: float = 2880
    k_f: float = 3.5e-4
    n_f: float = 0.874

    du1_xx = dde.grad.hessian(y, x, i=0, j=0)
    # TODO: check indices of jacobian
    du1_t = dde.grad.jacobian(y, x, i=0, j=1)

    u1 = y[..., 0].unsqueeze(1)

    # retardation_factor = 1 + (1 - por) / por * rho_s * k_f * torch.pow(u1, n_f - 1)
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u1 + 1e-6) ** (
        n_f - 1
    )

    return du1_t - D / retardation_factor * du1_xx
    
    
def pde_swe1d():
    raise NotImplementedError


def pde_swe2d(x, y):
    g = 1.0

    # non conservative form
    h_x = dde.grad.jacobian(y, x, i=0, j=0)
    h_y = dde.grad.jacobian(y, x, i=0, j=1)
    h_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_x = dde.grad.jacobian(y, x, i=1, j=0)
    u_y = dde.grad.jacobian(y, x, i=1, j=1)
    u_t = dde.grad.jacobian(y, x, i=1, j=2)
    v_x = dde.grad.jacobian(y, x, i=2, j=0)
    v_y = dde.grad.jacobian(y, x, i=2, j=1)
    v_t = dde.grad.jacobian(y, x, i=2, j=2)

    h = y[..., 0].unsqueeze(1)
    u = y[..., 1].unsqueeze(1)
    v = y[..., 2].unsqueeze(1)

    eq1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
    eq2 = u_t + u * u_x + v * u_y + g * h_x
    eq3 = v_t + u * v_x + v * v_y + g * h_y

    return eq1 + eq2 + eq3
