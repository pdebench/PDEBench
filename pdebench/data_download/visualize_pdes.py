import os
import argparse

from tqdm import tqdm
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
    num_samples = len(h5_file.keys())
    
    # randomly choose a seed for picking a sample that will subsequently be visualized 
    if not seed:
        seed = np.random.randint(0, num_samples) 

    # Ensure the seed number is defined
    assert seed < num_samples, "Seed number too high!"

    seed = str(seed).zfill(4)
    data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 1024, 1]

    h5_file.close()

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in tqdm(range(data.shape[0])):
        if i == 0:
           im = ax.plot(data[0].squeeze(), animated=True, color="blue")  # show an initial one first
        else:
           im = ax.plot(data[i].squeeze(), animated=True, color="blue")
        ax.plot
        ims.append([im[0]])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_diff_sorp.gif", writer=writer)
    print("Animation saved")


def visualize_2d_reacdiff(path, seed=None):
    """
    This function animates the 2D reaction-diffusion data.

    Args:
    path : path to the desired file
    seed : seed to select a specific sample/batch, ranging from 0-9999
    """

    # Read the h5 file and store the data
    h5_file = h5py.File(os.path.join(path, "2D_diff-react_NA_NA.h5"), "r")
    num_samples = len(h5_file.keys())

    # randomly choose a seed for picking a sample that will subsequently be visualized 
    if not seed:
        seed = np.random.randint(0, num_samples) 

    # Ensure the seed number is defined
    assert seed < num_samples, "Seed number too high!"

    seed = str(seed).zfill(4)
    data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 2]

    h5_file.close()

    # Initialize plot
    fig, ax = plt.subplots(1, 2)

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in tqdm(range(data.shape[0])):
        im1 = ax[0].imshow(data[i, ..., 0].squeeze(), animated=True)
        im2 = ax[1].imshow(data[i, ..., 1].squeeze(), animated=True)
        if i == 0:
            ax[0].imshow(data[0, ..., 0].squeeze())  # show an initial one first
            ax[1].imshow(data[0, ..., 1].squeeze())  # show an initial one first
        ims.append([im1, im2])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_2d_reacdiff.gif", writer=writer)
    print("Animation saved")


def visualize_swe(path, seed=None):
    """
    This function animates the shallow water equation data.

    Args:
    path : path to the desired file
    seed : seed to select a specific sample/batch, ranging from 0-9999
    """

    # Read the h5 file and store the data
    h5_file = h5py.File(os.path.join(path, "2D_rdb_NA_NA.h5"), "r")
    num_samples = len(h5_file.keys())
   
    # randomly choose a seed for picking a sample that will subsequently be visualized 
    if not seed:
        seed = np.random.randint(0, num_samples) 

    # Ensure the seed number is defined
    assert seed < num_samples, "Seed number too high!"

    seed = str(seed).zfill(4)
    data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 1]

    h5_file.close()

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in tqdm(range(data.shape[0])):
        im = ax.imshow(data[i].squeeze(), animated=True)
        if i == 0:
            ax.imshow(data[0].squeeze())  # show an initial one first
        ims.append([im])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_swe.gif", writer=writer)
    print("Animation saved")


def visualize_burgers(path, param=None):
    """
    This function animates the Burgers equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    # Read the h5 file and store the data
    if param is not None:
        flnm = "1D_Burgers_Sols_Nu" + str(param) + ".hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+path + flnm
    else:
        flnm = "1D_Burgers_Sols_Nu0.01.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        data = np.array(h5_file["tensor"], dtype=np.float32)[nb]  # (batch, t, x, channel) --> (t, x, channel)

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []

    for i in tqdm(range(data.shape[0])):
        if i == 0:
            im = ax.plot(xcrd, data[i].squeeze(), animated=True, color="blue")
        else:
            im = ax.plot(xcrd, data[i].squeeze(), animated=True, color="blue")  # show an initial one first
        ax.plot
        ims.append([im[0]])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_burgers.gif", writer=writer)
    print("Animation saved")


def visualize_advection(path, param=None):
    """
    This function animates the Advection equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    # Read the h5 file and store the data
    if param is not None:
        flnm = "1D_Advection_Sols_beta" + str(param) + ".hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+ path + flnm
    else:
        flnm = "1D_Advection_Sols_beta0.4.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        data = np.array(h5_file["tensor"], dtype=np.float32)[nb]  # (batch, t, x, channel) --> (t, x, channel)

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in tqdm(range(data.shape[0])):
        im = ax.plot(xcrd, data[i].squeeze(), animated=True)
        if i == 0:
            ax.plot(xcrd, data[i].squeeze())  # show an initial one first
        ax.plot
        ims.append([im[0]])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_advection.gif", writer=writer)
    print("Animation saved")


def visualize_1d_cfd(path, param=None):
    """
    This function animates the Burgers equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    # Read the h5 file and store the data
    if param is not None:
        assert len(param) == 4, 'param should include type,eta,zeta,boundary as list'
        flnm = "1D_CFD_" + str(param[0]) + "_Eta" + str(param[1]) + '_Zeta' + str(param[2]) +"_" + str(param[3]) + "_Train.hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+ path + flnm
    else:
        flnm = "1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        dd = np.array(h5_file["density"], dtype=np.float32)[nb]  # (batch, t, x, channel) --> (t, x, channel)

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    ax.set_title('density')
    for i in tqdm(range(dd.shape[0])):
        im = ax.plot(xcrd, dd[i].squeeze(), animated=True)
        if i == 0:
            ax.plot(xcrd, dd[i].squeeze())  # show an initial one first
        ax.plot
        ims.append([im[0]])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_1d_cfd.gif", writer=writer)
    print("Animation saved")


def visualize_2d_cfd(path, param=None):
    """
    This function animates the Burgers equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    # Read the h5 file and store the data
    if param is not None:
        assert len(param) == 6, 'param should include type,M,eta,zeta,boundary, resolution as list'
        flnm = "2D_CFD_" + str(param[0]) + "_M" + str(param[1]) + "_Eta" + str(param[2]) + '_Zeta' + str(param[3]) + "_" + str(param[4]) + "_" + str(param[5]) + "_Train.hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+ path + flnm
    else:
        flnm = "2D_CFD_Rand_M0.1_Eta1e-8_Zeta1e-8_periodic_512_Train.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        dd = np.array(h5_file["density"], dtype=np.float32)[nb]  # (batch, t, x, y, channel) --> (t, x, y, channel)

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    ax.set_title('density')
    for i in range(dd.shape[0]):
        im = ax.imshow(dd[i].squeeze(), animated=True)
        ims.append([im])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie.gif", writer=writer)
    print("saved")


def visualize_3d_cfd(path, param=None):
    # Read the h5 file and store the data
    if param is not None:
        assert len(param) == 5, 'param should include type,M,eta,zeta,boundary as list'
        flnm = "3D_CFD_" + str(param[0]) + "_M" + str(param[1]) + "_Eta" + str(param[2]) + '_Zeta' + str(param[3]) + "_" + str(param[4]) + "_Train.hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+ path + flnm
    else:
        flnm = "3D_CFD_Rand_M1.0_Eta1e-8_Zeta1e-8_periodic_Train.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        dd = np.array(h5_file["density"], dtype=np.float32)[nb]  # (batch, t, x, y, channel) --> (t, x, y, channel)

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    ax.set_title('density')
    for i in range(dd.shape[0]):
        im = ax.imshow(dd[i, :, :, 32].squeeze(), animated=True)
        ims.append([im])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie.gif", writer=writer)
    print("saved")


def visualize_ns_incom():
    pass


def visualize_darcy(path, param=None):
    """
    This function animates Darcy Flow equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    # Read the h5 file and store the data
    if param is not None:
        flnm = "2D_DarcyFlow_beta" + str(param) + "_Train.hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+ path + flnm
    else:
        flnm = "2D_DarcyFlow_beta1.0_Train.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        data = np.array(h5_file["tensor"], dtype=np.float32)[nb]  # (batch, t, x, y, channel) --> (t, x, y, channel)
        nu = np.array(h5_file["nu"], dtype=np.float32)[nb]  # (batch, t, x, y, channel) --> (t, x, y, channel)

    # Initialize plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(data.squeeze())
    ax[1].imshow(nu.squeeze())
    ax[0].set_title('Data u')
    ax[1].set_title('diffusion coefficient nu')
    plt.savefig('2D_DarcyFlow.pdf')
    print("plot saved")


def visualize_1d_reacdiff(path, param=None):
    """
    This function animates 1D Reaction Diffusion equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    # Read the h5 file and store the data
    if param is not None:
        assert len(param) == 2, 'param should include Nu and Rho as list'
        flnm = "ReacDiff_Nu" + str(param[0]) + '_Rho' + str(param[1]) +".hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+ path + flnm
    else:
        flnm = "ReacDiff_Nu1.0_Rho1.0.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        data = np.array(h5_file["tensor"], dtype=np.float32)[nb]  # (batch, t, x, channel) --> (t, x, channel)

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in tqdm(range(data.shape[0])):
        im = ax.plot(xcrd, data[i].squeeze(), animated=True)
        if i == 0:
            ax.plot(xcrd, data[i].squeeze())  # show an initial one first
        ax.plot
        ims.append([im[0]])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_1d_reacdiff.gif", writer=writer)
    print("Animation saved")


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
    arg_parser.add_argument(
        "--param",
        type=float,
        default=None,
        help="PDE parameter to be plotted",
    )
    arg_parser.add_argument(
        "--params",
        nargs='+',
        default=None,
        help="PDE parameters to be plotted",
    )

    args = arg_parser.parse_args()

    if args.pde_name == "diff_sorp":
        visualize_diff_sorp(args.data_path, args.seed_number)
    elif args.pde_name == "2d_reacdiff":
        visualize_2d_reacdiff(args.data_path, args.seed_number)
    elif args.pde_name == "swe":
        visualize_swe(args.data_path, args.seed_number)
    elif args.pde_name == "burgers":
        visualize_burgers(args.data_path, args.param)
    elif args.pde_name == "advection":
        visualize_advection(args.data_path, args.param)
    elif args.pde_name == "1d_cfd":
        visualize_1d_cfd(args.data_path, args.params)
    elif args.pde_name == "2d_cfd":
        visualize_2d_cfd(args.data_path, args.params)
    elif args.pde_name == "3d_cfd":
        visualize_3d_cfd(args.data_path, args.params)
    elif args.pde_name == "darcy":
        visualize_darcy(args.data_path, args.param)
    elif args.pde_name == "1d_reacdiff":
        visualize_1d_reacdiff(args.data_path, args.params)
    else:
        raise ValueError("PDE name not recognized!")

