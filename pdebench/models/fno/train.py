from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from pdebench.models.fno.fno import FNO1d, FNO2d, FNO3d
from pdebench.models.fno.utils import FNODatasetMult, FNODatasetSingle
from pdebench.models.metrics import metrics
from torch import nn

# torch.manual_seed(0)
# np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    if_training,
    continue_training,
    num_workers,
    modes,
    width,
    initial_step,
    t_train,
    num_channels,
    batch_size,
    epochs,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    model_update,
    flnm,
    single_file,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    plot,
    channel_plot,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    base_path="../data/",
    training_type="autoregressive",
):
    # print(
    #    f"Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}"
    # )

    ################################################################
    # load data
    ################################################################

    if single_file:
        # filename
        model_name = flnm[:-5] + "_FNO"
        # print("FNODatasetSingle")

        # Initialize the dataset and dataloader
        train_data = FNODatasetSingle(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            saved_folder=base_path,
        )
        val_data = FNODatasetSingle(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            if_test=True,
            saved_folder=base_path,
        )

    else:
        # filename
        model_name = flnm + "_FNO"

        # print("FNODatasetMult")
        train_data = FNODatasetMult(
            flnm,
            saved_folder=base_path,
        )
        val_data = FNODatasetMult(
            flnm,
            if_test=True,
            saved_folder=base_path,
        )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    ################################################################
    # training and evaluation
    ################################################################

    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    # print("Spatial Dimension", dimensions - 3)
    if dimensions == 4:
        model = FNO1d(
            num_channels=num_channels,
            width=width,
            modes=modes,
            initial_step=initial_step,
        ).to(device)
    elif dimensions == 5:
        model = FNO2d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            initial_step=initial_step,
        ).to(device)
    elif dimensions == 6:
        model = FNO3d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            modes3=modes,
            initial_step=initial_step,
        ).to(device)

    # Set maximum time step of the data to train
    t_train = min(t_train, _data.shape[-2])

    model_path = model_name + ".pt"

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters = {total_params}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.inf

    start_epoch = 0

    if not if_training:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        errs = metrics(
            val_loader,
            model,
            Lx,
            Ly,
            Lz,
            plot,
            channel_plot,
            model_name,
            x_min,
            x_max,
            y_min,
            y_max,
            t_min,
            t_max,
            initial_step=initial_step,
        )
        with Path(model_name + ".pickle").open("wb") as pb:
            pickle.dump(errs, pb)

        return

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        # print("Restoring model (that is the network's weights) from file...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()

        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]
        loss_val_min = checkpoint["loss"]

    for ep in range(start_epoch, epochs):
        model.train()
        # t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy, grid in train_loader:
            loss = 0

            # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            xx = xx.to(device)  # noqa: PLW2901
            yy = yy.to(device)  # noqa: PLW2901
            grid = grid.to(device)  # noqa: PLW2901

            # Initialize the prediction tensor
            pred = yy[..., :initial_step, :]
            # Extract shape of the input tensor for reshaping (i.e. stacking the
            # time and channels dimension together)
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            if training_type in ["autoregressive"]:
                # Autoregressive loop
                for t in range(initial_step, t_train):
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)

                    # Extract target at current time step
                    y = yy[..., t : t + 1, :]

                    # Model run
                    im = model(inp, grid)

                    # Loss calculation
                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)

                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901

                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]  # if t_train is not -1
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if training_type in ["single"]:
                x = xx[..., 0, :]
                y = yy[..., t_train - 1 : t_train, :]
                pred = model(x, grid)
                _batch = yy.size(0)
                loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                train_l2_step += loss.item()
                train_l2_full += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if ep % model_update == 0:
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                for xx, yy, grid in val_loader:
                    loss = 0
                    xx = xx.to(device)  # noqa: PLW2901
                    yy = yy.to(device)  # noqa: PLW2901
                    grid = grid.to(device)  # noqa: PLW2901

                    if training_type in ["autoregressive"]:
                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)

                        for t in range(initial_step, yy.shape[-2]):
                            inp = xx.reshape(inp_shape)
                            y = yy[..., t : t + 1, :]
                            im = model(inp, grid)
                            _batch = im.size(0)
                            loss += loss_fn(
                                im.reshape(_batch, -1), y.reshape(_batch, -1)
                            )

                            pred = torch.cat((pred, im), -2)

                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901

                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]
                        val_l2_full += loss_fn(
                            _pred.reshape(_batch, -1), _yy.reshape(_batch, -1)
                        ).item()

                    if training_type in ["single"]:
                        x = xx[..., 0, :]
                        y = yy[..., t_train - 1 : t_train, :]
                        pred = model(x, grid)
                        _batch = yy.size(0)
                        loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                        val_l2_step += loss.item()
                        val_l2_full += loss.item()

                if val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    torch.save(
                        {
                            "epoch": ep,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss_val_min,
                        },
                        model_path,
                    )

        # t2 = default_timer()
        scheduler.step()
        # print(
        #    "epoch: {0}, loss: {1:.5f}, t2-t1: {2:.5f}, trainL2: {3:.5f}, testL2: {4:.5f}".format(
        #        ep, loss.item(), t2 - t1, train_l2_full, val_l2_full
        #    )
        # )


if __name__ == "__main__":
    run_training()
    # print("Done.")
