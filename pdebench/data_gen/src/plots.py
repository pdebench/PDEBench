"""
Author : John Kim, Simon Brown, Timothy Praditia
PDE Simulation packages
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
import phi.vis as phivis
import os


def plot_data(data, t, dim, channel, t_fraction, config, filename):
    t_idx = int(t_fraction * (data.shape[0] - 1))

    # Plot data at t=t_idx, use imshow for 2D data
    plt.figure()
    plt.title(f"$t={t[t_idx]}$")
    if dim == 1:
        x = np.array(h5_file["grid"]["x"], dtype="f")
        plt.plot(x.squeeze(), data[t_idx, ..., channel])
        plt.xlabel("$x$")
    else:
        plt.imshow(
            data[t_idx, ..., channel].transpose(),
            aspect="auto",
            origin="lower",
            extent=[
                config.sim.x_left,
                config.sim.x_right,
                config.sim.y_bottom,
                config.sim.y_top,
            ],
        )
        plt.xlabel("$x$")
        plt.ylabel("$y$")
    plt.tight_layout()
    plt.savefig(filename)


def save_phi_plot(result, title, filepath, bbox_inches='tight', pad_inches=0):
    """
    save one custom figure from an array
    """
    phivis.plot(result)
    plt.title(title)
    plt.savefig(filepath, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close()


def phi_plots(results, T_results, title, filepath, scale = 1, bbox_inches='tight', pad_inches=0):
    """
    Save simulation custom figures, get images list
    """
    images = []
    upperfilepath = filepath
    for i, arr in enumerate(T_results):
        filename = '{}.png'.format(title)
        if upperfilepath == '':
            filepath = filename
        else:
            filepath = upperfilepath + '/{}'.format(filename)
        save_phi_plot(
            scale * results[i], title, filepath, bbox_inches=bbox_inches, pad_inches=pad_inches)
        images.append(imageio.imread(filepath))
    return images


def save_sim_figures(results, T_results, simulation_name, kinematic_value, filepath, scale = 1, bbox_inches='tight', pad_inches=0):
    """
    save figures, get images list
    """
    images = []
    upperfilepath = filepath
    for i, arr in enumerate(T_results):
        res = arr[0]
        title = '{}_{}_t={}'.format(simulation_name, kinematic_value, round(T_results[i], 2))
        filename = '{}.png'.format(title)
        if upperfilepath == '':
            filepath = filename
        else:
            filepath = upperfilepath + '/{}'.format(filename)
        save_phi_plot(
            scale * res, title, filepath, bbox_inches=bbox_inches, pad_inches=pad_inches)
        images.append(imageio.imread(filepath))
    return images


def save_gif(paths, images, DT=0.1):
    """
    Saving images into .gif animation
    """
    imageio.mimsave(
        paths,
        images,
        duration=DT,
    )
