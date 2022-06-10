"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import torch

from typing import Tuple

sys.path.append(".")
from .utils import (
    PINNDatasetRadialDambreak,
    PINNDatasetDiffReact,
    PINNDataset2D,
    PINNDatasetDiffSorption,
    PINNDatasetBump,
)
from .pde_definitions import (
    pde_diffusion_reaction,
    pde_swe2d,
    pde_diffusion_sorption,
    pde_swe1d,
)

from metrics import metrics, metric_func


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


def setup_swe_2d(filename, seed) -> Tuple[dde.Model, PINNDataset2D]:

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


def run_training(scenario, epochs, learning_rate, model_update, flnm, seed):
    if scenario == "swe2d":
        model, dataset = setup_swe_2d(filename=flnm, seed=seed)
        n_components = 1
    elif scenario == "diff-react":
        model, dataset = setup_diffusion_reaction(filename=flnm, seed=seed)
        n_components = 2
    elif scenario == "diff-sorp":
        model, dataset = setup_diffusion_sorption(filename=flnm, seed=seed)
        n_components = 1
    else:
        raise NotImplementedError(f"PINN training not implemented for {scenario}")

    # filename
    model_name = flnm + "_PINN"

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

    errs = metric_func(test_pred, test_gt)
    errors = [np.array(err.cpu()) for err in errs]
    print(errors)
    pickle.dump(errors, open(model_name + ".pickle", "wb"))

    # plot sample
    plot_input = dataset.generate_plot_input(time=1.0)
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

    # TODO: implement function to get specific timestep from dataset
    # y_true = dataset[:][1][-xdim * ydim :]

    # print("L2 relative error:", dde.metrics.l2_relative_error(y_true.cpu(), y_pred))


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
        flnm="2D_diff-react_NA_NA_0000.h5",
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
