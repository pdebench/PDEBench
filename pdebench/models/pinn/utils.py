# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:43:15 2022

@author: timot
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import h5py
from omegaconf import DictConfig, OmegaConf
import yaml


class PINNDataset1D(Dataset):
    def __init__(self, filename, seed):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """
        self.seed = seed

        # load data file
        root_path = os.path.abspath("../data")
        data_path = os.path.join(root_path, filename)
        with h5py.File(data_path, "r") as h5_file:
            seed_group = h5_file[seed]

            # extract config
            self.config = yaml.load(seed_group.attrs["config"], Loader=yaml.SafeLoader)

            # build input data from individual dimensions
            # dim x = [x]
            self.data_grid_x = torch.tensor(seed_group["grid"]["x"], dtype=torch.float)
            # # dim t = [t]
            self.data_grid_t = torch.tensor(seed_group["grid"]["t"], dtype=torch.float)

            XX, TT = torch.meshgrid(
                [self.data_grid_x, self.data_grid_t],
                indexing="ij",
            )

            self.data_input = torch.vstack([XX.ravel(), TT.ravel()]).T

            self.data_output = torch.tensor(
                np.array(seed_group["data"]), dtype=torch.float
            )

            # permute from [t, x] -> [x, t]
            permute_idx = list(range(1, len(self.data_output.shape) - 1))
            permute_idx.extend(list([0, -1]))
            self.data_output = self.data_output.permute(permute_idx)

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_t))
        test_input_t = self.data_input[:, 1].reshape((n_x, n_t))
        test_output = self.data_output.reshape((n_x, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, -n_last_time_steps:]
        test_input_t = test_input_t[:, -n_last_time_steps:]
        test_output = test_output[:, -n_last_time_steps:, :]

        test_input = torch.vstack([test_input_x.ravel(), test_input_t.ravel()]).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked.T

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        return raveled_tensor.reshape((1, n_x, n_last_time_steps, n_components))

    def generate_plot_input(self, time=1.0):
        x_space = np.linspace(
            self.config["sim"]["x_left"],
            self.config["sim"]["x_right"],
            self.config["sim"]["xdim"],
        )
        # xx, yy = np.meshgrid(x_space, y_space)

        tt = np.ones_like(x_space) * time
        val_input = np.vstack((x_space, tt)).T
        return val_input

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx].unsqueeze(1)

    def get_name(self):
        return self.config["name"]


class PINNDataset2D(Dataset):
    def __init__(self, filename, seed):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """
        self.seed = seed

        # load data file
        root_path = os.path.abspath("../data")
        data_path = os.path.join(root_path, filename)
        with h5py.File(data_path, "r") as h5_file:
            seed_group = h5_file[seed]

            # extract config
            self.config = yaml.load(seed_group.attrs["config"], Loader=yaml.SafeLoader)

            # build input data from individual dimensions
            # dim x = [x]
            self.data_grid_x = torch.tensor(seed_group["grid"]["x"], dtype=torch.float)
            # # dim y = [y]
            self.data_grid_y = torch.tensor(seed_group["grid"]["y"], dtype=torch.float)
            # # dim t = [t]
            self.data_grid_t = torch.tensor(seed_group["grid"]["t"], dtype=torch.float)

            XX, YY, TT = torch.meshgrid(
                [self.data_grid_x, self.data_grid_y, self.data_grid_t],
                indexing="ij",
            )

            self.data_input = torch.vstack([XX.ravel(), YY.ravel(), TT.ravel()]).T

            self.data_output = torch.tensor(
                np.array(seed_group["data"]), dtype=torch.float
            )

            # permute from [t, x, y] -> [x, y, t]
            permute_idx = list(range(1, len(self.data_output.shape) - 1))
            permute_idx.extend(list([0, -1]))
            self.data_output = self.data_output.permute(permute_idx)

    def generate_plot_input(self, time=1.0):
        x_space = np.linspace(
            self.config["sim"]["x_left"],
            self.config["sim"]["x_right"],
            self.config["sim"]["xdim"],
        )
        y_space = np.linspace(
            self.config["sim"]["y_bottom"],
            self.config["sim"]["y_top"],
            self.config["sim"]["ydim"],
        )
        xx, yy = np.meshgrid(x_space, y_space)
        tt = np.ones_like(xx) * time
        val_input = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(tt))).T
        return val_input

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx].unsqueeze(1)

    def get_name(self):
        return self.config["name"]

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_y = len(self.data_grid_y)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_y, n_t))
        test_input_y = self.data_input[:, 1].reshape((n_x, n_y, n_t))
        test_input_t = self.data_input[:, 2].reshape((n_x, n_y, n_t))
        test_output = self.data_output.reshape((n_x, n_y, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, :, -n_last_time_steps:]
        test_input_y = test_input_y[:, :, -n_last_time_steps:]
        test_input_t = test_input_t[:, :, -n_last_time_steps:]
        test_output = test_output[:, :, -n_last_time_steps:, :]

        test_input = torch.vstack(
            [test_input_x.ravel(), test_input_y.ravel(), test_input_t.ravel()]
        ).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked.T

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_y = len(self.data_grid_y)
        return raveled_tensor.reshape((1, n_x, n_y, n_last_time_steps, n_components))


class PINNDatasetBump(PINNDataset1D):
    def __init__(self, filename, seed):
        super().__init__(filename, seed)

        # ravel data
        self.data_output = self.data_output.ravel()


class PINNDatasetRadialDambreak(PINNDataset2D):
    def __init__(self, filename, seed):
        super().__init__(filename, seed)

        # ravel data
        self.data_output = self.data_output.ravel()

    def get_initial_condition_func(self):
        # TODO: gather parameters from dataset
        def initial_h(coords):
            x0 = 0.0
            y0 = 0.0
            x = coords[:, 0]
            y = coords[:, 1]
            r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            h_in = 2.0
            h_out = 1.0
            dam_radius = self.config["sim"]["dam_radius"]

            h_initial = np.expand_dims(
                h_in * (r <= dam_radius) + h_out * (r > dam_radius), 1
            )

            return h_initial

        return initial_h


class PINNDatasetDiffReact(PINNDataset2D):
    def __init__(self, filename, seed):
        super().__init__(filename, seed)

        # TODO: find way to ravel multiple components in parent class
        self.data_u = self.data_output[:, :, :, 0].ravel()
        self.data_v = self.data_output[:, :, :, 1].ravel()

    def get_initial_condition(self):
        Nx = len(self.data_grid_x)
        Ny = len(self.data_grid_y)
        Nt = len(self.data_grid_t)

        np.random.seed(self.config["sim"]["seed"])

        u0 = np.random.randn(Nx * Ny)
        v0 = np.random.randn(Nx * Ny)

        u0 = u0.reshape(Nx * Ny)
        v0 = v0.reshape(Nx * Ny)

        x_space = np.linspace(
            self.config["sim"]["x_left"],
            self.config["sim"]["x_right"],
            self.config["sim"]["xdim"],
        )
        y_space = np.linspace(
            self.config["sim"]["y_bottom"],
            self.config["sim"]["y_top"],
            self.config["sim"]["ydim"],
        )
        xx, yy = np.meshgrid(self.data_grid_x.cpu(), self.data_grid_y.cpu())
        tt = np.zeros_like(xx)
        ic_input = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(tt))).T

        return (
            ic_input,
            np.expand_dims(u0, 1),
            np.expand_dims(v0, 1),
        )

    def __getitem__(self, idx):
        return (
            self.data_input[idx, :],
            self.data_u[idx].unsqueeze(1),
            self.data_v[idx].unsqueeze(1),
        )


class PINNDatasetDiffSorption(PINNDataset1D):
    def __init__(self, filename, seed):
        super().__init__(filename, seed)

        # ravel data
        self.data_output = self.data_output.ravel()

    def get_initial_condition(self):
        # Generate initial condition
        Nx = self.config["sim"]["xdim"]

        np.random.seed(self.config["sim"]["seed"])

        u0 = np.ones(Nx) * np.random.uniform(0, 0.2)

        return (self.data_input[:Nx, :], np.expand_dims(u0, 1))

class PINNDataset1Dpde(Dataset):
    def __init__(self, filename, root_path='data', val_batch_idx=-1):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """

        # load data file
        data_path = os.path.join(root_path, filename)
        h5_file = h5py.File(data_path, "r")

        # build input data from individual dimensions
        # dim x = [x]
        self.data_grid_x = torch.tensor(h5_file["x-coordinate"], dtype=torch.float)
        self.dx = self.data_grid_x[1] - self.data_grid_x[0]
        self.xL = self.data_grid_x[0] - 0.5 * self.dx
        self.xR = self.data_grid_x[-1] + 0.5 * self.dx
        self.xdim = self.data_grid_x.size(0)
        # # dim t = [t]
        self.data_grid_t = torch.tensor(h5_file["t-coordinate"], dtype=torch.float)

        # main data
        keys = list(h5_file.keys())
        keys.sort()
        if 'tensor' in keys:
            self.data_output = torch.tensor(np.array(h5_file["tensor"][val_batch_idx]),
                                            dtype=torch.float)
            # permute from [t, x] -> [x, t]
            self.data_output = self.data_output.T

            # for init/boundary conditions
            self.init_data = self.data_output[..., 0, None]
            self.bd_data_L = self.data_output[0, :, None]
            self.bd_data_R = self.data_output[-1, :, None]

        else:
            _data1 = np.array(h5_file["density"][val_batch_idx])
            _data2 = np.array(h5_file["Vx"][val_batch_idx])
            _data3 = np.array(h5_file["pressure"][val_batch_idx])
            _data = np.concatenate([_data1[...,None], _data2[...,None], _data3[...,None]], axis=-1)
            # permute from [t, x] -> [x, t]
            _data = np.transpose(_data, (1, 0, 2))

            self.data_output = torch.tensor(_data, dtype=torch.float)
            del(_data, _data1, _data2, _data3)

            # for init/boundary conditions
            self.init_data = self.data_output[:, 0]
            self.bd_data_L = self.data_output[0]
            self.bd_data_R = self.data_output[-1]

        self.tdim = self.data_output.size(1)
        self.data_grid_t = self.data_grid_t[:self.tdim]

        XX, TT = torch.meshgrid(
            [self.data_grid_x, self.data_grid_t],
            indexing="ij",
        )

        self.data_input = torch.vstack([XX.ravel(), TT.ravel()]).T

        h5_file.close()
        if 'tensor' in keys:
            self.data_output = self.data_output.reshape(-1, 1)
        else:
            self.data_output = self.data_output.reshape(-1, 3)

    def get_initial_condition(self):
        # return (self.data_grid_x[:, None], self.init_data)
        return (self.data_input[::self.tdim, :], self.init_data)

    def get_boundary_condition(self):
        # return (self.data_grid_t[:self.nt, None], self.bd_data_L, self.bd_data_R)
        return (self.data_input[:self.xdim, :], self.bd_data_L, self.bd_data_R)

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_t))
        test_input_t = self.data_input[:, 1].reshape((n_x, n_t))
        test_output = self.data_output.reshape((n_x, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, -n_last_time_steps:]
        test_input_t = test_input_t[:, -n_last_time_steps:]
        test_output = test_output[:, -n_last_time_steps:, :]

        test_input = torch.vstack([test_input_x.ravel(), test_input_t.ravel()]).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked.T

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        return raveled_tensor.reshape((1, n_x, n_last_time_steps, n_components))

    def generate_plot_input(self, time=1.0):
        x_space = np.linspace(self.xL, self.xR, self.xdim)
        # xx, yy = np.meshgrid(x_space, y_space)

        tt = np.ones_like(x_space) * time
        val_input = np.vstack((x_space, tt)).T
        return val_input

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx]

class PINNDataset2Dpde(Dataset):
    def __init__(self, filename, root_path='data', val_batch_idx=-1, rdc_x=9, rdc_y=9):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """

        # load data file
        data_path = os.path.join(root_path, filename)
        h5_file = h5py.File(data_path, "r")

        # build input data from individual dimensions
        # dim x = [x]
        self.data_grid_x = torch.tensor(h5_file["x-coordinate"], dtype=torch.float)
        self.data_grid_x = self.data_grid_x[::rdc_x]
        self.dx = self.data_grid_x[1] - self.data_grid_x[0]
        self.xL = self.data_grid_x[0] - 0.5 * self.dx
        self.xR = self.data_grid_x[-1] + 0.5 * self.dx
        self.xdim = self.data_grid_x.size(0)
        # dim y = [y]
        self.data_grid_y = torch.tensor(h5_file["y-coordinate"], dtype=torch.float)
        self.data_grid_y = self.data_grid_y[::rdc_y]
        self.dy = self.data_grid_y[1] - self.data_grid_y[0]
        self.yL = self.data_grid_y[0] - 0.5 * self.dy
        self.yR = self.data_grid_y[-1] + 0.5 * self.dy
        self.ydim = self.data_grid_y.size(0)
        # # dim t = [t]
        self.data_grid_t = torch.tensor(h5_file["t-coordinate"], dtype=torch.float)

        # main data
        _data1 = np.array(h5_file["density"][val_batch_idx])
        _data2 = np.array(h5_file["Vx"][val_batch_idx])
        _data3 = np.array(h5_file["Vy"][val_batch_idx])
        _data4 = np.array(h5_file["pressure"][val_batch_idx])
        _data = np.concatenate([_data1[...,None], _data2[...,None], _data3[...,None], _data4[...,None]],
                               axis=-1)
        # permute from [t, x, y, v] -> [x, y, t, v]
        _data = np.transpose(_data, (1, 2, 0, 3))
        _data = _data[::rdc_x, ::rdc_y]

        self.data_output = torch.tensor(_data, dtype=torch.float)
        del(_data, _data1, _data2, _data3, _data4)

        # for init/boundary conditions
        self.init_data = self.data_output[..., 0, :]
        self.bd_data_xL = self.data_output[0]
        self.bd_data_xR = self.data_output[-1]
        self.bd_data_yL = self.data_output[:,0]
        self.bd_data_yR = self.data_output[:,-1]

        self.tdim = self.data_output.size(2)
        self.data_grid_t = self.data_grid_t[:self.tdim]

        XX, YY, TT = torch.meshgrid(
            [self.data_grid_x, self.data_grid_y, self.data_grid_t],
            indexing="ij",
        )

        self.data_input = torch.vstack([XX.ravel(), YY.ravel(), TT.ravel()]).T

        h5_file.close()
        self.data_output = self.data_output.reshape(-1, 4)

    def get_initial_condition(self):
        # return (self.data_grid_x[:, None], self.init_data)
        return (self.data_input[::self.tdim, :], self.init_data)

    def get_boundary_condition(self):
        # return (self.data_grid_t[:self.nt, None], self.bd_data_L, self.bd_data_R)
        return (self.data_input[:self.xdim*self.ydim, :],
                self.bd_data_xL, self.bd_data_xR,
                self.bd_data_yL, self.bd_data_yR
                )

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_y = len(self.data_grid_y)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_y, n_t))
        test_input_y = self.data_input[:, 1].reshape((n_x, n_y, n_t))
        test_input_t = self.data_input[:, 4].reshape((n_x, n_y, n_t))
        test_output = self.data_output.reshape((n_x, n_y, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, :, -n_last_time_steps:]
        test_input_y = test_input_y[:, :, -n_last_time_steps:]
        test_input_t = test_input_t[:, :, -n_last_time_steps:]
        test_output = test_output[:, :, -n_last_time_steps:, :]

        test_input = torch.vstack(
            [test_input_x.ravel(), test_input_y.ravel(), test_input_t.ravel()]
        ).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked.T

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_y = len(self.data_grid_y)
        return raveled_tensor.reshape((1, n_x, n_y, n_last_time_steps, n_components))

    def generate_plot_input(self, time=1.0):
        return None

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx].unsqueeze(1)

class PINNDataset3Dpde(Dataset):
    def __init__(self, filename, root_path='data', val_batch_idx=-1, rdc_x=2, rdc_y=2, rdc_z=2):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """

        # load data file
        data_path = os.path.join(root_path, filename)
        h5_file = h5py.File(data_path, "r")

        # build input data from individual dimensions
        # dim x = [x]
        self.data_grid_x = torch.tensor(h5_file["x-coordinate"], dtype=torch.float)
        self.data_grid_x = self.data_grid_x[::rdc_x]
        self.dx = self.data_grid_x[1] - self.data_grid_x[0]
        self.xL = self.data_grid_x[0] - 0.5 * self.dx
        self.xR = self.data_grid_x[-1] + 0.5 * self.dx
        self.xdim = self.data_grid_x.size(0)
        # dim y = [y]
        self.data_grid_y = torch.tensor(h5_file["y-coordinate"], dtype=torch.float)
        self.data_grid_y = self.data_grid_y[::rdc_y]
        self.dy = self.data_grid_y[1] - self.data_grid_y[0]
        self.yL = self.data_grid_y[0] - 0.5 * self.dy
        self.yR = self.data_grid_y[-1] + 0.5 * self.dy
        self.ydim = self.data_grid_y.size(0)
        # dim z = [z]
        self.data_grid_z = torch.tensor(h5_file["z-coordinate"], dtype=torch.float)
        self.data_grid_z = self.data_grid_z[::rdc_z]
        self.dz = self.data_grid_z[1] - self.data_grid_z[0]
        self.zL = self.data_grid_z[0] - 0.5 * self.dz
        self.zR = self.data_grid_z[-1] + 0.5 * self.dz
        self.zdim = self.data_grid_z.size(0)
        # # dim t = [t]
        self.data_grid_t = torch.tensor(h5_file["t-coordinate"], dtype=torch.float)

        # main data
        _data1 = np.array(h5_file["density"][val_batch_idx])
        _data2 = np.array(h5_file["Vx"][val_batch_idx])
        _data3 = np.array(h5_file["Vy"][val_batch_idx])
        _data4 = np.array(h5_file["Vz"][val_batch_idx])
        _data5 = np.array(h5_file["pressure"][val_batch_idx])
        _data = np.concatenate([_data1[...,None], _data2[...,None], _data3[...,None], _data4[...,None], _data5[...,None]],
                               axis=-1)
        # permute from [t, x, y, z, v] -> [x, y, z, t, v]
        _data = np.transpose(_data, (1, 2, 3, 0, 4))
        _data = _data[::rdc_x, ::rdc_y, ::rdc_z]

        self.data_output = torch.tensor(_data, dtype=torch.float)
        del(_data, _data1, _data2, _data3, _data4, _data5)

        # for init/boundary conditions
        self.init_data = self.data_output[..., 0, :]
        self.bd_data_xL = self.data_output[0]
        self.bd_data_xR = self.data_output[-1]
        self.bd_data_yL = self.data_output[:,0]
        self.bd_data_yR = self.data_output[:,-1]
        self.bd_data_zL = self.data_output[:,:,0]
        self.bd_data_zR = self.data_output[:,:,-1]

        self.tdim = self.data_output.size(3)
        self.data_grid_t = self.data_grid_t[:self.tdim]

        XX, YY, ZZ, TT = torch.meshgrid(
            [self.data_grid_x, self.data_grid_y, self.data_grid_z, self.data_grid_t],
            indexing="ij",
        )

        self.data_input = torch.vstack([XX.ravel(), YY.ravel(), ZZ.ravel(), TT.ravel()]).T

        h5_file.close()
        self.data_output = self.data_output.reshape(-1, 5)

    def get_initial_condition(self):
        # return (self.data_grid_x[:, None], self.init_data)
        return (self.data_input[::self.tdim, :], self.init_data)

    def get_boundary_condition(self):
        # return (self.data_grid_t[:self.nt, None], self.bd_data_L, self.bd_data_R)
        return (self.data_input[:self.xdim*self.ydim*self.zdim, :],
                self.bd_data_xL, self.bd_data_xR,
                self.bd_data_yL, self.bd_data_yR,
                self.bd_data_zL, self.bd_data_zR,
                )

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_y = len(self.data_grid_y)
        n_z = len(self.data_grid_z)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_y, n_z, n_t))
        test_input_y = self.data_input[:, 1].reshape((n_x, n_y, n_z, n_t))
        test_input_z = self.data_input[:, 2].reshape((n_x, n_y, n_z, n_t))
        test_input_t = self.data_input[:, 3].reshape((n_x, n_y, n_z, n_t))
        test_output = self.data_output.reshape((n_x, n_y, n_z, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, :, :, -n_last_time_steps:]
        test_input_y = test_input_y[:, :, :, -n_last_time_steps:]
        test_input_z = test_input_z[:, :, :, -n_last_time_steps:]
        test_input_t = test_input_t[:, :, :, -n_last_time_steps:]
        test_output = test_output[:, :, :, -n_last_time_steps:, :]

        test_input = torch.vstack(
            [test_input_x.ravel(), test_input_y.ravel(), test_input_z.ravel(), test_input_t.ravel()]
        ).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked.T

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_y = len(self.data_grid_y)
        n_z = len(self.data_grid_z)
        return raveled_tensor.reshape((1, n_x, n_y, n_z, n_last_time_steps, n_components))

    def generate_plot_input(self, time=1.0):
        return None

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx].unsqueeze(1)