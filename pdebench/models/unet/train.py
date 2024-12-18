from __future__ import annotations

import logging
import pickle
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from pdebench.models.metrics import metrics
from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d
from pdebench.models.unet.utils import UNetDatasetMult, UNetDatasetSingle
from torch import nn

# torch.manual_seed(0)
# np.random.seed(0)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    if_training,
    continue_training,
    num_workers,
    initial_step,
    t_train,
    in_channels,
    out_channels,
    batch_size,
    unroll_step,
    ar_mode,
    pushforward,
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
    msg = f"Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}"
    logger.info(msg)

    ################################################################
    # load data
    ################################################################

    if single_file:
        # filename
        model_name = flnm[:-5] + "_Unet"

        # Initialize the dataset and dataloader
        train_data = UNetDatasetSingle(
            flnm,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
        )
        val_data = UNetDatasetSingle(
            flnm,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            if_test=True,
        )

    else:
        # filename
        model_name = flnm + "_Unet"

        train_data = UNetDatasetMult(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            saved_folder=base_path,
        )
        val_data = UNetDatasetMult(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
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

    # model = UNet2d(in_channels, out_channels).to(device)
    _, _data = next(iter(val_loader))
    dimensions = len(_data.shape)
    msg = f"Spatial Dimension: {dimensions - 3}"
    logger.info(msg)
    if training_type in ["autoregressive"]:
        if dimensions == 4:
            model = UNet1d(in_channels * initial_step, out_channels).to(device)
        elif dimensions == 5:
            model = UNet2d(in_channels * initial_step, out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(in_channels * initial_step, out_channels).to(device)
    if training_type in ["single"]:
        if dimensions == 4:
            model = UNet1d(in_channels, out_channels).to(device)
        elif dimensions == 5:
            model = UNet2d(in_channels, out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(in_channels, out_channels).to(device)

    # Set maximum time step of the data to train
    t_train = min(t_train, _data.shape[-2])
    # Set maximum of unrolled time step for the pushforward trick
    if t_train - unroll_step < 1:
        unroll_step = t_train - 1

    if training_type in ["autoregressive"]:
        if ar_mode:
            if pushforward:
                model_name = model_name + "-PF-" + str(unroll_step)
            if not pushforward:
                unroll_step = _data.shape[-2]
                model_name = model_name + "-AR"
        else:
            model_name = model_name + "-1-step"

    model_path = model_name + ".pt"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = f"Total parameters = {total_params}"
    logger.info(msg)

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
            mode="Unet",
            initial_step=initial_step,
        )
        pickle.dump(errs, Path.open(model_name + ".pickle", "wb"))

        return

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        msg = "Restoring model (that is the network's weights) from file..."
        logger.info(msg)
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

    msg = "start training..."
    logger.info(msg)

    if ar_mode:
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0

            for xx, yy in train_loader:
                loss = 0

                # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
                # yy: target tensor [b, x1, ..., xd, t, v]
                # grid: meshgrid [b, x1, ..., xd, dims]
                xx_tensor = xx.to(device)
                yy_tensor = yy.to(device)

                if training_type in ["autoregressive"]:
                    # Initialize the prediction tensor
                    pred = yy[..., :initial_step, :]

                    # Extract shape of the input tensor for reshaping (i.e. stacking the
                    # time and channels dimension together)
                    inp_shape = list(xx_tensor.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    # Autoregressive loop
                    for t in range(initial_step, t_train):
                        if t < t_train - unroll_step:
                            with torch.no_grad():
                                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                                inp = xx_tensor.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                                inp = inp.permute(temp_shape)

                                # Extract target at current time step
                                y = yy_tensor[..., t : t + 1, :]

                                # Model run
                                temp_shape = [0]
                                temp_shape.extend(list(range(2, len(inp.shape))))
                                temp_shape.append(1)
                                im = model(inp).permute(temp_shape).unsqueeze(-2)

                                # Concatenate the prediction at current time step into the
                                # prediction tensor
                                pred = torch.cat((pred, im), -2)

                                # Concatenate the prediction at the current time step to be used
                                # as input for the next time step
                                xx_tensor = torch.cat(
                                    (xx_tensor[..., 1:, :], im), dim=-2
                                )

                        else:
                            # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                            inp = xx_tensor.reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                            inp = inp.permute(temp_shape)

                            # Extract target at current time step
                            y = yy_tensor[..., t : t + 1, :]

                            # Model run
                            temp_shape = [0]
                            temp_shape.extend(list(range(2, len(inp.shape))))
                            temp_shape.append(1)
                            im = model(inp).permute(temp_shape).unsqueeze(-2)

                            # Loss calculation
                            loss += loss_fn(
                                im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                            )

                            # Concatenate the prediction at current time step into the
                            # prediction tensor
                            pred = torch.cat((pred, im), -2)

                            # Concatenate the prediction at the current time step to be used
                            # as input for the next time step
                            xx_tensor = torch.cat((xx_tensor[..., 1:, :], im), dim=-2)

                    train_l2_step += loss.item()
                    _batch = yy_tensor.size(0)
                    _yy = yy_tensor[..., :t_train, :]
                    l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                    train_l2_full += l2_full.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if training_type in ["single"]:
                x = xx[..., 0, :]
                y = yy[..., t_train - 1 : t_train, :]
                pred = model(x.permute([0, 2, 1])).permute([0, 2, 1])
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
                    for xx, yy in val_loader:
                        loss = 0
                        xx_tensor = xx.to(device)
                        yy_tensor = yy.to(device)

                        if training_type in ["autoregressive"]:
                            pred = yy_tensor[..., :initial_step, :]
                            inp_shape = list(xx.shape)
                            inp_shape = inp_shape[:-2]
                            inp_shape.append(-1)

                            for t in range(initial_step, t_train):
                                inp = xx_tensor.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                                inp = inp.permute(temp_shape)
                                y = yy_tensor[..., t : t + 1, :]
                                temp_shape = [0]
                                temp_shape.extend(list(range(2, len(inp.shape))))
                                temp_shape.append(1)
                                im = model(inp).permute(temp_shape).unsqueeze(-2)
                                loss += loss_fn(
                                    im.reshape(batch_size, -1),
                                    y.reshape(batch_size, -1),
                                )

                                pred = torch.cat((pred, im), -2)

                                xx_tensor = torch.cat(
                                    (xx_tensor[..., 1:, :], im), dim=-2
                                )

                            val_l2_step += loss.item()
                            _batch = yy.size(0)
                            _pred = pred[..., initial_step:t_train, :]
                            _yy = yy_tensor[..., initial_step:t_train, :]
                            val_l2_full += loss_fn(
                                _pred.reshape(_batch, -1), _yy.reshape(_batch, -1)
                            ).item()

                    if training_type in ["single"]:
                        x = xx[..., 0, :]
                        y = yy[..., t_train - 1 : t_train, :]
                        pred = model(x.permute([0, 2, 1])).permute([0, 2, 1])
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

            t2 = default_timer()
            scheduler.step()
            msg = f"epoch: {ep}, loss: {loss.item():.5f}, t2-t1: {t2 - t1:.5f}, trainL2: {train_l2_step:.5f}, testL2: {val_l2_step:.5f}"
            logger.info(msg)

    else:
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0

            for xx, yy in train_loader:
                loss = 0

                # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
                # yy: target tensor [b, x1, ..., xd, t, v]
                xx_tensor = xx.to(device)
                yy_tensor = yy.to(device)

                # Initialize the prediction tensor
                pred = yy_tensor[..., :initial_step, :]

                # Extract shape of the input tensor for reshaping (i.e. stacking the
                # time and channels dimension together)
                inp_shape = list(xx_tensor.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                # Autoregressive loop
                for t in range(initial_step, t_train):
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = yy_tensor[..., t - initial_step : t, :].reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                    inp = inp.permute(temp_shape)
                    inp = torch.normal(inp, 0.001)

                    # Extract target at current time step
                    y = yy_tensor[..., t : t + 1, :]

                    # Model run
                    temp_shape = [0]
                    temp_shape.extend(list(range(2, len(inp.shape))))
                    temp_shape.append(1)
                    im = model(inp).permute(temp_shape).unsqueeze(-2)

                    # Loss calculation
                    loss += loss_fn(
                        im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                    )

                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)

                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    # xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy_tensor[..., :t_train, :]  # if t_train is not -1
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ep % model_update == 0 or ep == epochs:
                val_l2_step = 0
                val_l2_full = 0
                with torch.no_grad():
                    for xx, yy in val_loader:
                        loss = 0
                        xx_tensor = xx.to(device)
                        yy_tensor = yy.to(device)

                        pred = yy_tensor[..., :initial_step, :]
                        inp_shape = list(xx_tensor.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)

                        for t in range(initial_step, t_train):
                            inp = yy_tensor[..., t - initial_step : t, :].reshape(
                                inp_shape
                            )
                            temp_shape = [0, -1]
                            temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                            inp = inp.permute(temp_shape)
                            y = yy_tensor[..., t : t + 1, :]
                            temp_shape = [0]
                            temp_shape.extend(list(range(2, len(inp.shape))))
                            temp_shape.append(1)
                            im = model(inp).permute(temp_shape).unsqueeze(-2)
                            loss += loss_fn(
                                im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                            )

                            pred = torch.cat((pred, im), -2)

                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        # if t_train is not -1
                        _yy = yy_tensor[..., initial_step:t_train, :]
                        val_l2_full += loss_fn(
                            _pred.reshape(_batch, -1), _yy.reshape(_batch, -1)
                        ).item()

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

            t2 = default_timer()
            scheduler.step()
            msg = f"epoch: {ep}, loss: {loss.item():.5f}, t2-t1: {t2 - t1:.5f}, trainL2: {train_l2_step:.5f}, testL2: {val_l2_step:.5f}"
            logger.info(msg)


if __name__ == "__main__":
    run_training()
    msg = "Done."
    logger.info(msg)
