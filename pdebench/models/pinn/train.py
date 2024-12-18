"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

from __future__ import annotations

import pickle
from pathlib import Path

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
from pdebench.models.metrics import metric_func
from pdebench.models.pinn.pde_definitions import (
    pde_adv1d,
    pde_burgers1D,
    pde_CFD1d,
    pde_CFD2d,
    pde_diffusion_reaction,
    pde_diffusion_reaction_1d,
    pde_diffusion_sorption,
    pde_swe2d,
)
from pdebench.models.pinn.utils import (
    PINNDataset1Dpde,
    PINNDataset2D,
    PINNDataset2Dpde,
    PINNDataset3Dpde,
    PINNDatasetDiffReact,
    PINNDatasetDiffSorption,
    PINNDatasetRadialDambreak,
)


def setup_diffusion_sorption(filename, seed):
    # TODO: read from dataset config file
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    D = 5e-4

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    bc_d = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 1.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    )

    def operator_bc(inputs, outputs, X):
        # compute u_t
        du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return outputs - D * du_x

    bc_d2 = dde.icbc.OperatorBC(
        geomtime,
        operator_bc,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
    )

    dataset = PINNDatasetDiffSorption(filename, seed)

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1])

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_sorption,
        [ic, bc_d, bc_d2, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")

    def transform_output(x, y):
        return torch.relu(y)

    net.apply_output_transform(transform_output)

    model = dde.Model(data, net)

    return model, dataset


def setup_diffusion_reaction(filename, seed):
    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0, 5.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

    dataset = PINNDatasetDiffReact(filename, seed)
    initial_input, initial_u, initial_v = dataset.get_initial_condition()

    ic_data_u = dde.icbc.PointSetBC(initial_input, initial_u, component=0)
    ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_reaction,
        [bc, ic_data_u, ic_data_v, bc_data_u, bc_data_v],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset


def setup_swe_2d(filename, seed) -> tuple[dde.Model, PINNDataset2D]:
    dataset = PINNDatasetRadialDambreak(filename, seed)

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-2.5, -2.5), (2.5, 2.5))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic_h = dde.icbc.IC(
        geomtime,
        dataset.get_initial_condition_func(),
        lambda _, on_initial: on_initial,
        component=0,
    )
    ic_u = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=1
    )
    ic_v = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=2
    )

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde_swe2d,
        [bc, ic_h, ic_u, ic_v, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset


def _boundary_r(x, on_boundary, xL, xR):
    return (on_boundary and np.isclose(x[0], xL)) or (
        on_boundary and np.isclose(x[0], xR)
    )


def setup_pde1D(
    filename="1D_Advection_Sols_beta0.1.hdf5",
    root_path="data",
    val_batch_idx=-1,
    input_ch=2,
    output_ch=1,
    hidden_ch=40,
    xL=0.0,
    xR=1.0,
    if_periodic_bc=True,
    aux_params=None,
):
    # TODO: read from dataset config file
    if aux_params is None:
        aux_params = [0.1]
    geom = dde.geometry.Interval(xL, xR)
    boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, xL, xR)
    if filename[0] == "R":
        timedomain = dde.geometry.TimeDomain(0, 1.0)
        pde = lambda x, y: pde_diffusion_reaction_1d(x, y, aux_params[0], aux_params[1])
    elif filename.split("_")[1][0] == "A":
        timedomain = dde.geometry.TimeDomain(0, 2.0)
        pde = lambda x, y: pde_adv1d(x, y, aux_params[0])
    elif filename.split("_")[1][0] == "B":
        timedomain = dde.geometry.TimeDomain(0, 2.0)
        pde = lambda x, y: pde_burgers1D(x, y, aux_params[0])
    elif filename.split("_")[1][0] == "C":
        timedomain = dde.geometry.TimeDomain(0, 1.0)
        pde = lambda x, y: pde_CFD1d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset1Dpde(
        filename, root_path=root_path, val_batch_idx=val_batch_idx
    )
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    if filename.split("_")[1][0] == "C":
        ic_data_d = dde.icbc.PointSetBC(
            initial_input.cpu(), initial_u[:, 0].unsqueeze(1), component=0
        )
        ic_data_v = dde.icbc.PointSetBC(
            initial_input.cpu(), initial_u[:, 1].unsqueeze(1), component=1
        )
        ic_data_p = dde.icbc.PointSetBC(
            initial_input.cpu(), initial_u[:, 2].unsqueeze(1), component=2
        )
    else:
        ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u, component=0)
    # prepare boundary condition
    if if_periodic_bc:
        if filename.split("_")[1][0] == "C":
            bc_D = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            bc_V = dde.icbc.PeriodicBC(geomtime, 1, boundary_r)
            bc_P = dde.icbc.PeriodicBC(geomtime, 2, boundary_r)

            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_d, ic_data_v, ic_data_p, bc_D, bc_V, bc_P],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
        else:
            bc = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_u, bc],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
    else:
        ic = dde.icbc.IC(
            geomtime,
            lambda x: -np.sin(np.pi * x[:, 0:1]),
            lambda _, on_initial: on_initial,
        )
        bd_input, bd_uL, bd_uR = dataset.get_boundary_condition()
        bc_data_uL = dde.icbc.PointSetBC(bd_input.cpu(), bd_uL, component=0)
        bc_data_uR = dde.icbc.PointSetBC(bd_input.cpu(), bd_uR, component=0)

        data = dde.data.TimePDE(
            geomtime,
            pde,
            [ic, bc_data_uL, bc_data_uR],
            num_domain=1000,
            num_boundary=1000,
            num_initial=5000,
        )
    net = dde.nn.FNN(
        [input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal"
    )
    model = dde.Model(data, net)

    return model, dataset


def setup_CFD2D(
    filename="2D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
    root_path="data",
    val_batch_idx=-1,
    input_ch=2,
    output_ch=4,
    hidden_ch=40,
    xL=0.0,
    xR=1.0,
    yL=0.0,
    yR=1.0,
    if_periodic_bc=True,
    aux_params=None,
):
    # TODO: read from dataset config file
    if aux_params is None:
        aux_params = [1.6667]
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset2Dpde(
        filename, root_path=root_path, val_batch_idx=val_batch_idx
    )
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 0].unsqueeze(1), component=0
    )
    ic_data_vx = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 1].unsqueeze(1), component=1
    )
    ic_data_vy = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 2].unsqueeze(1), component=2
    )
    ic_data_p = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 3].unsqueeze(1), component=3
    )
    # prepare boundary condition
    # bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_p],  # , bc],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN(
        [input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal"
    )
    model = dde.Model(data, net)

    return model, dataset


def setup_CFD3D(
    filename="3D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
    root_path="data",
    val_batch_idx=-1,
    input_ch=2,
    output_ch=4,
    hidden_ch=40,
    aux_params=None,
):
    # TODO: read from dataset config file
    if aux_params is None:
        aux_params = [1.6667]
    geom = dde.geometry.Cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset3Dpde(
        filename, root_path=root_path, val_batch_idx=val_batch_idx
    )
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 0].unsqueeze(1), component=0
    )
    ic_data_vx = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 1].unsqueeze(1), component=1
    )
    ic_data_vy = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 2].unsqueeze(1), component=2
    )
    ic_data_vz = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 3].unsqueeze(1), component=3
    )
    ic_data_p = dde.icbc.PointSetBC(
        initial_input.cpu(), initial_u[..., 4].unsqueeze(1), component=4
    )
    # prepare boundary condition
    bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_vz, ic_data_p, bc],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN(
        [input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal"
    )
    model = dde.Model(data, net)

    return model, dataset


def _run_training(
    scenario,
    epochs,
    learning_rate,
    model_update,
    flnm,
    input_ch,
    output_ch,
    root_path,
    val_batch_idx,
    if_periodic_bc,
    aux_params,
    if_single_run,
    seed,
):
    if scenario == "swe2d":
        model, dataset = setup_swe_2d(filename=flnm, seed=seed)
        n_components = 1
    elif scenario == "diff-react":
        model, dataset = setup_diffusion_reaction(filename=flnm, seed=seed)
        n_components = 2
    elif scenario == "diff-sorp":
        model, dataset = setup_diffusion_sorption(filename=flnm, seed=seed)
        n_components = 1
    elif scenario == "pde1D":
        model, dataset = setup_pde1D(
            filename=flnm,
            root_path=root_path,
            input_ch=input_ch,
            output_ch=output_ch,
            val_batch_idx=val_batch_idx,
            if_periodic_bc=if_periodic_bc,
            aux_params=aux_params,
        )
        n_components = 3 if flnm.split("_")[1][0] == "C" else 1
    elif scenario == "CFD2D":
        model, dataset = setup_CFD2D(
            filename=flnm,
            root_path=root_path,
            input_ch=input_ch,
            output_ch=output_ch,
            val_batch_idx=val_batch_idx,
            aux_params=aux_params,
        )
        n_components = 4
    elif scenario == "CFD3D":
        model, dataset = setup_CFD3D(
            filename=flnm,
            root_path=root_path,
            input_ch=input_ch,
            output_ch=output_ch,
            val_batch_idx=val_batch_idx,
            aux_params=aux_params,
        )
        n_components = 5
    else:
        msg = f"PINN training not implemented for {scenario}"
        raise NotImplementedError(msg)

    # filename
    model_name = flnm + "_PINN" if if_single_run else flnm[:-5] + "_PINN"

    checker = dde.callbacks.ModelCheckpoint(
        f"{model_name}.pt", save_better_only=True, period=5000
    )

    model.compile("adam", lr=learning_rate)
    losshistory, train_state = model.train(
        epochs=epochs, display_every=model_update, callbacks=[checker]
    )

    test_input, test_gt = dataset.get_test_data(
        n_last_time_steps=20, n_components=n_components
    )
    # select only n_components of output
    # dirty hack for swe2d where we predict more components than we have data on
    test_pred = torch.tensor(model.predict(test_input.cpu())[:, :n_components])

    # prepare data for metrics eval
    test_pred = dataset.unravel_tensor(
        test_pred, n_last_time_steps=20, n_components=n_components
    )
    test_gt = dataset.unravel_tensor(
        test_gt, n_last_time_steps=20, n_components=n_components
    )

    if if_single_run:
        errs = metric_func(test_pred, test_gt)
        errors = [np.array(err.cpu()) for err in errs]
        with Path(model_name + ".pickle").open("wb") as f:
            pickle.dump(errors, f)

        # plot sample
        plot_input = dataset.generate_plot_input(time=1.0)
        if scenario == "pde1D":
            xdim = dataset.xdim
            dim = 1
        else:
            dim = dataset.config["plot"]["dim"]
            xdim = dataset.config["sim"]["xdim"]
            if dim == 2:
                ydim = dataset.config["sim"]["ydim"]

        y_pred = model.predict(plot_input)[:, 0]
        if dim == 1:
            plt.figure()
            plt.plot(y_pred)
        elif dim == 2:
            im_data = y_pred.reshape(xdim, ydim)
            plt.figure()
            plt.imshow(im_data)

        plt.savefig(f"{model_name}.png")
        return None

        # TODO: implement function to get specific timestep from dataset
        # y_true = dataset[:][1][-xdim * ydim :]
    return test_pred, test_gt, model_name


def run_training(
    scenario,
    epochs,
    learning_rate,
    model_update,
    flnm,
    input_ch=1,
    output_ch=1,
    root_path="../data/",
    val_num=10,
    if_periodic_bc=True,
    aux_params=None,
    seed="0000",
):
    if aux_params is None:
        aux_params = [None]
    if val_num == 1:  # single job
        _run_training(
            scenario,
            epochs,
            learning_rate,
            model_update,
            flnm,
            input_ch,
            output_ch,
            root_path,
            -val_num,
            if_periodic_bc,
            aux_params,
            if_single_run=True,
            seed=seed,
        )
    else:
        for val_batch_idx in range(-1, -val_num, -1):
            test_pred, test_gt, model_name = _run_training(
                scenario,
                epochs,
                learning_rate,
                model_update,
                flnm,
                input_ch,
                output_ch,
                root_path,
                val_batch_idx,
                if_periodic_bc,
                aux_params,
                if_single_run=False,
                seed=seed,
            )
            if val_batch_idx == -1:
                pred, target = test_pred.unsqueeze(0), test_gt.unsqueeze(0)
            else:
                pred = torch.cat([pred, test_pred.unsqueeze(0)], 0)
                target = torch.cat([target, test_gt.unsqueeze(0)], 0)

        errs = metric_func(test_pred, test_gt)
        errors = [np.array(err.cpu()) for err in errs]
        with Path(model_name + ".pickle").open("wb") as f:
            pickle.dump(errors, f)


if __name__ == "__main__":
    # run_training(
    #     scenario="diff-sorp",
    #     epochs=100,
    #     learning_rate=1e-3,
    #     model_update=500,
    #     flnm="2D_diff-sorp_NA_NA_0000.h5",
    #     seed="0000",
    # )
    run_training(
        scenario="diff-react",
        epochs=100,
        learning_rate=1e-3,
        model_update=500,
        flnm="2D_diff-react_NA_NA.h5",
        seed="0000",
    )
    # run_training(
    #     scenario="swe2d",
    #     epochs=100,
    #     learning_rate=1e-3,
    #     model_update=500,
    #     flnm="radial_dam_break_0000.h5",
    #     seed="0000",
    # )
