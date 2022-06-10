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
