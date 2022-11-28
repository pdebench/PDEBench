import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


pdes = (
    "advection",
    "burgers",
    "1d_cfd",
    "diff_sorp",
    "1d_reacdiff",
    "2d_cfd",
    "darcy",
    "2d_reacdiff",
    "ns_incom",
    "swe",
    "3d_cfd",
)


def visualize_diff_sorp(path, seed=None):
    """
    This function animates the diffusion sorption data.

    Args:
    path : path to the desired file
    seed : seed to select a specific sample/batch, ranging from 0-9999
    """

    # Read the h5 file and store the data
    h5_file = h5py.File(os.path.join(path, "1D_diff-sorp_NA_NA.h5"), "r")

    if not seed:
        seed = np.random.randint(0, len(h5_file.keys()))

    # Ensure the seed number is defined
    assert seed < data.shape[0], "Seed number too high!"

    seed = str(seed).zfill(4)
    data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 1024, 1]

    h5_file.close()

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in range(data.shape[0]):
        im = ax.plot(data[i].squeeze(), animated=True)
        if i == 0:
            ax.plot(data[0].squeeze())  # show an initial one first
        ax.plot
        ims.append([im])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie.gif", writer=writer)
    print("saved")


def visualize_2d_reacdiff(path, seed=None):
    """
    This function animates the 2D reaction-diffusion data.

    Args:
    path : path to the desired file
    seed : seed to select a specific sample/batch, ranging from 0-9999
    """

    # Read the h5 file and store the data
    h5_file = h5py.File(os.path.join(path, "2D_diff-react_NA_NA.h5"), "r")

    if not seed:
        seed = np.random.randint(0, len(h5_file.keys()))

    # Ensure the seed number is defined
    assert seed < data.shape[0], "Seed number too high!"

    seed = str(seed).zfill(4)
    data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 2]

    h5_file.close()

    # Initialize plot
    fig, ax = plt.subplots(1, 2)

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in range(data.shape[0]):
        im1 = ax[0].imshow(data[i, ..., 0].squeeze(), animated=True)
        im2 = ax[1].imshow(data[i, ..., 1].squeeze(), animated=True)
        if i == 0:
            ax[0].imshow(data[0, ..., 0].squeeze())  # show an initial one first
            ax[1].imshow(data[0, ..., 1].squeeze())  # show an initial one first
        ims.append([im1, im2])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie.gif", writer=writer)
    print("saved")


def visualize_swe(path, seed=None):
    """
    This function animates the shallow water equation data.

    Args:
    path : path to the desired file
    seed : seed to select a specific sample/batch, ranging from 0-9999
    """

    # Read the h5 file and store the data
    h5_file = h5py.File(os.path.join(path, "2D_rdb_NA_NA.h5"), "r")

    if not seed:
        seed = np.random.randint(0, len(h5_file.keys()))

    # Ensure the seed number is defined
    assert seed < data.shape[0], "Seed number too high!"

    seed = str(seed).zfill(4)
    data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 1]

    h5_file.close()

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in range(data.shape[0]):
        im = ax.imshow(data[i].squeeze(), animated=True)
        if i == 0:
            ax.imshow(data[0].squeeze())  # show an initial one first
        ims.append([im])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie.gif", writer=writer)
    print("saved")


def visualize_burgers():
    pass


def visualize_advection():
    pass


def visualize_1d_cfd():
    pass


def visualize_2d_cfd():
    pass


def visualize_3d_cfd():
    pass


def visualize_ns_incom():
    pass


def visualize_darcy():
    pass


def visualize_1d_reacdiff():
    pass


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Visualize PDEs",
        description="Helper script to visualize PDEs in the PDEBench dataset",
        epilog="",
    )

    arg_parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="Path to the hdf5 data where the downloaded data reside",
    )
    arg_parser.add_argument(
        "--pde_name", type=str, help="Name of the PDE dataset to download"
    )
    arg_parser.add_argument(
        "--seed_number",
        type=int,
        default=None,
        help="Seed number to define which sample/batch to be plotted",
    )

    args = arg_parser.parse_args()

    if args.pde_name == "diff_sorp":
        visualize_diff_sorp(args.data_path, args.seed_number)
    elif args.pde_name == "2d_reacdiff":
        visualize_2d_reacdiff(args.data_path, args.seed_number)
    elif args.pde_name == "swe":
        visualize_swe(args.data_path, args.seed_number)
    else:
        raise ValueError("PDE name not recognized!")
