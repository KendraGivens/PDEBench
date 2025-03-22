from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def plot_data_single_time(data, t, dim, channel, t_fraction, config, filename):
    """
    Plot a single time step (fraction of total).
    """
    t_idx = int(t_fraction * (data.shape[0] - 1))
    plt.figure()
    plt.title(f"t={t[t_idx]:.3f}")

    if dim == 1:
        # 1D plot
        plt.plot(data[t_idx, ..., channel])
        plt.xlabel("x")
    else:
        # 2D imshow
        plt.imshow(
            data[t_idx, ..., channel].T,
            origin="lower",
            aspect="auto",
            extent=[
                config.sim.x_left,
                config.sim.x_right,
                config.sim.y_bottom,
                config.sim.y_top,
            ],
            cmap="viridis"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="Field Value")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_data_all_time_steps(data, t, dim, channel, config, output_prefix):
    """
    Plot *every* time step in 'data' and save each as a separate image.
    """
    num_timesteps = data.shape[0]
    for t_idx in range(num_timesteps):
        plt.figure()
        plt.title(f"t={t[t_idx]:.3f}")
        if dim == 1:
            plt.plot(data[t_idx, ..., channel])
            plt.xlabel("x")
        else:
            plt.imshow(
                data[t_idx, ..., channel].T,
                origin="lower",
                aspect="auto",
                extent=[
                    config.sim.x_left,
                    config.sim.x_right,
                    config.sim.y_bottom,
                    config.sim.y_top,
                ],
                cmap="viridis"
            )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar(label="Field Value")

        plt.tight_layout()
        filename = f"{output_prefix}_t{t_idx:04d}.png"
        plt.savefig(filename)
        plt.close()


def plot_data_subplots(data, t, dim, channel, config, output_prefix):
    """
    Plot a few time steps in a single multi-panel figure.
    E.g., 4 frames spaced evenly through the simulation.
    """
    num_timesteps = data.shape[0]
    n_frames = config.plot.frames  # how many subplots you want
    # pick frames spaced evenly
    indices = np.round(np.linspace(0, num_timesteps - 1, n_frames)).astype(int)

    fig, axs = plt.subplots(1, n_frames, figsize=(5*n_frames, 5))
    for i, idx in enumerate(indices):
        ax = axs[i] if n_frames > 1 else axs
        ax.set_title(f"t={t[idx]:.3f}")
        if dim == 1:
            ax.plot(data[idx, ..., channel])
            ax.set_xlabel("x")
        else:
            im = ax.imshow(
                data[idx, ..., channel].T,
                origin="lower",
                aspect="auto",
                extent=[
                    config.sim.x_left,
                    config.sim.x_right,
                    config.sim.y_bottom,
                    config.sim.y_top,
                ],
                cmap="viridis"
            )
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_subplots.png")
    plt.close()


def plot_data_gif(data, t, dim, channel, config, output_prefix):
    """
    Create an animated GIF for all time steps.
    """
    num_timesteps = data.shape[0]
    temp_files = []

    # 1) Generate each frame as a PNG
    for t_idx in range(num_timesteps):
        filename = f"{output_prefix}_frame_{t_idx:04d}.png"
        temp_files.append(filename)

        plt.figure()
        plt.title(f"t={t[t_idx]:.3f}")
        if dim == 1:
            plt.plot(data[t_idx, ..., channel])
            plt.xlabel("x")
        else:
            plt.imshow(
                data[t_idx, ..., channel].T,
                origin="lower",
                aspect="auto",
                extent=[
                    config.sim.x_left,
                    config.sim.x_right,
                    config.sim.y_bottom,
                    config.sim.y_top,
                ],
                cmap="viridis"
            )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar(label="Field Value")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # 2) Combine into a GIF
    images = [imageio.imread(fname) for fname in temp_files]
    gif_filename = f"{output_prefix}.gif"
    imageio.mimsave(gif_filename, images, duration=config.plot.duration)

    # 3) (Optional) Cleanup the PNG files after creating the GIF
    for fname in temp_files:
        os.remove(fname)


@hydra.main(config_path="configs/", config_name="constant_water_inflow")
def main(config: DictConfig):
    
    # Hydra changes working directory; revert to original
    os.chdir(get_original_cwd())

    # Build full path to the HDF5 file
    work_path = Path(config.work_dir)
    output_path = work_path / config.data_dir / config.output_path
    config.output_path = output_path / config.output_path
    data_path = config.output_path.with_suffix(".h5")


    # Open HDF5 file
    with h5py.File(data_path, "r") as h5_file:
        groups = [g for g in h5_file.keys() if isinstance(h5_file[g], h5py.Group)]

        # If you specify config.sim.seed, we only plot that seed. Otherwise plot all.
        if "seed" in config.sim:
            seeds_to_plot = [str(config.sim.seed).zfill(4)]
        else:
            seeds_to_plot = groups

        for seed_str in seeds_to_plot:
            data = np.array(h5_file[f"{seed_str}/data"], dtype="f")
            t    = np.array(h5_file[f"{seed_str}/grid/t"], dtype="f")

            # Prepare prefix for output files
            output_prefix = f"{config.name}_{seed_str}"

            # Check which mode is requested
            mode = config.plot.mode.lower()
            if mode == "single":
                # Single time step (the fraction is config.plot.t_idx)
                filename = f"{output_prefix}_single.png"
                plot_data_single_time(
                    data,
                    t,
                    config.plot.dim,
                    config.plot.channel_idx,
                    config.plot.t_idx,
                    config,
                    filename
                )
            elif mode == "all_time_steps":
                # One image per time step
                plot_data_all_time_steps(
                    data,
                    t,
                    config.plot.dim,
                    config.plot.channel_idx,
                    config,
                    output_prefix
                )
            elif mode == "subplots":
                # Multiple time steps in a single figure
                plot_data_subplots(
                    data,
                    t,
                    config.plot.dim,
                    config.plot.channel_idx,
                    config,
                    output_prefix
                )
            elif mode == "gif":
                # Create an animated GIF across all time steps
                plot_data_gif(
                    data,
                    t,
                    config.plot.dim,
                    config.plot.channel_idx,
                    config,
                    output_prefix
                )
            else:
                print(f"Unknown plot mode: {mode}")
            break

    print("Finished plotting!")


if __name__ == "__main__":
    main()