import matplotlib.pyplot as plt
import matplotlib.animation as animation

from trajectories.trajectory_utils import *
from trajectories.constants import *


def plot_static_2D_trajectory(pedestrian, boundaries=None, show=True, save_path=None):
    """Plot the trajectory of a pedestrian, as an image

    Parameters
    ----------
    pedestrian : obj
        A pedestrian
    boundaries : obj, optional
        The boundaries of the environment to be use as axis limits, by default None
    show : bool, optional
        Whether or not the image should be displayed, by default True
    save_path : str, optional
        The path to the file where the image will be saved, by default None
    """
    x, y = pedestrian.get_trajectory_column("x"), pedestrian.get_trajectory_column("y")
    plt.scatter(x / 1000, y / 1000, c="cornflowerblue", s=10)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    if boundaries:
        plt.xlim([boundaries["xmin"] / 1000, boundaries["xmax"] / 1000])
        plt.ylim([boundaries["ymin"] / 1000, boundaries["ymax"] / 1000])
    plt.title(f"Trajectory of pedestrian {pedestrian.ped_id}")
    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)


def plot_animated_2D_trajectory(
    pedestrian, boundaries=None, show=True, save_path=None, loop=False
):
    """Plot the trajectory of a pedestrian, as an animation

    Parameters
    ----------
    pedestrian : obj
        A pedestrian
    boundaries : obj, optional
        The boundaries of the environment to be use as axis limits, by default None
    show : bool, optional
        Whether or not the animation should be displayed, by default True
    save_path : str, optional
        The path to the file where the animation will be saved, by default None
    loop: bool, optional
        Whether or not the animation should loop, by default False
    """
    x, y = pedestrian.get_trajectory_column("x"), pedestrian.get_trajectory_column("y")

    colors = ["cornflowerblue"] * len(x)

    fig, ax = plt.subplots()
    ax.scatter([], [], c=[])  # plot of x and y in time

    if boundaries:
        xmin, xmax = boundaries["xmin"] / 1000, boundaries["xmax"] / 1000
        ymin, ymax = boundaries["ymin"] / 1000, boundaries["ymax"] / 1000
    else:
        xmin, xmax = min(x) / 1000, max(x) / 1000
        ymin, ymax = min(y) / 1000, max(y) / 1000

    def animate(i, ax, pos_Nx, pos_Ny, colorsN):
        ax.clear()
        ax.scatter(pos_Nx[:i], pos_Ny[:i], c=colorsN[:i], s=10)
        ax.set_title(f"Trajectory of {pedestrian.ped_id}")
        ax.axis([xmin, xmax, ymin, ymax])

    ax.axis([xmin, xmax, ymin, ymax])

    ax.set_title(f"Trajectory of {pedestrian.ped_id}")
    ax.set_autoscale_on(False)

    ani = animation.FuncAnimation(
        fig,
        animate,
        range(len(x)),
        fargs=(ax, x / 1000, y / 1000, colors),
        repeat=loop,
        interval=50,
        blit=False,
    )

    if show:
        plt.show()

    if save_path:
        ani.save(
            save_path,
            writer="imagemagick",
            fps=100,
        )


def plot_static_2D_trajectories(
    pedestrians,
    simultaneous=False,
    boundaries=None,
    colors=None,
    title=None,
    show=True,
    save_path=None,
):
    """Plot the trajectories of a set of pedestrians

    Parameters
    ----------
    pedestrians : list
        A list of pedestrians
    simultaneous : bool, optional
        Whether or not the trajectories should be cropped to the simultaneous observations, by default False
    boundaries : obj, optional
        The boundaries of the environment to be used as axis limits, by default None
    colors : list, optional
        A list of colors to be used, by default None
    title : str, optional
        The title of the figure, by default None
    show : bool, optional
        Whether or not the image should be displayed, by default True
    save_path : str, optional
        The path to the file where the image will be saved, by default None
    """
    n_ped = len(pedestrians)
    ped_ids = [ped.ped_id for ped in pedestrians]

    if simultaneous:
        trajectories = compute_simultaneous_observations(
            [ped.get_trajectory() for ped in pedestrians]
        )
    else:
        trajectories = [ped.get_trajectory() for ped in pedestrians]

    if title is None:
        title = f"Trajectories for {'-'.join([str(ped.ped_id) for ped in pedestrians])}"

    if not colors:
        if n_ped > len(COLORS):
            colors = np.random.choice(COLORS, n_ped)
        else:
            colors = COLORS[:n_ped]

    for ped_id, trajectory, color in zip(ped_ids, trajectories, colors):
        x, y = trajectory[:, 1], trajectory[:, 2]
        plt.scatter(x / 1000, y / 1000, c=color, s=10, label=ped_id)

    if boundaries:
        plt.xlim([boundaries["xmin"] / 1000, boundaries["xmax"] / 1000])
        plt.ylim([boundaries["ymin"] / 1000, boundaries["ymax"] / 1000])

    plt.title(title)
    plt.xlabel("x (m)")
    plt.legend()

    plt.ylabel("y (m)")
    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)


def plot_animated_2D_trajectories(
    pedestrians,
    simultaneous=False,
    boundaries=None,
    colors=None,
    title=None,
    show=True,
    save_path=None,
    loop=False,
):
    """Plot the trajectory of a set of pedestrians, as an animation

    Parameters
    ----------
    pedestrians : list
        A list of pedestrians
    simultaneous : bool, optional
        Whether or not the trajectories should be cropped to the simultaneous observations, by default False
    boundaries : obj, optional
        The boundaries of the environment to be use as axis limits, by default None
    show : bool, optional
        Whether or not the animation should be displayed, by default True
    save_path : str, optional
        The path to the file where the animation will be saved, by default None
    loop: bool, optional
        Whether or not the animation should loop, by default False
    """
    n_ped = len(pedestrians)
    ped_ids = [ped.ped_id for ped in pedestrians]

    if simultaneous:
        trajectories = compute_simultaneous_observations(
            [ped.get_trajectory() for ped in pedestrians]
        )
    else:
        trajectories = get_padded_trajectories(
            [ped.get_trajectory() for ped in pedestrians]
        )

    positions = [traj[:, 1:3] / 1000 for traj in trajectories]

    if title is None:
        title = f"Trajectories for {'-'.join([str(ped.ped_id) for ped in pedestrians])}"

    if not colors:
        if n_ped > len(COLORS):
            colors = np.random.choice(COLORS, n_ped)
        else:
            colors = COLORS[:n_ped]

    fig, ax = plt.subplots()
    ax.scatter([], [], c=[])  # plot of x and y in time

    if boundaries:
        xmin, xmax = boundaries["xmin"] / 1000, boundaries["xmax"] / 1000
        ymin, ymax = boundaries["ymin"] / 1000, boundaries["ymax"] / 1000
    else:
        xmin, xmax = (
            min([np.nanmin(pos[:, 0]) for pos in positions]),
            max([np.nanmax(pos[:, 0]) for pos in positions]),
        )
        ymin, ymax = (
            min([np.nanmin(pos[:, 1]) for pos in positions]),
            max([np.nanmax(pos[:, 1]) for pos in positions]),
        )

    def animate(i, ax, positions, colors, title):
        ax.clear()
        for position, color in zip(positions, colors):
            print(position)
            ax.scatter(position[:i, 0], position[:i, 1], c=color, s=10)
        ax.set_title(title)
        ax.axis([xmin, xmax, ymin, ymax])

    ax.axis([xmin, xmax, ymin, ymax])

    ax.set_title(title)
    ax.set_autoscale_on(False)

    ani = animation.FuncAnimation(
        fig,
        animate,
        range(len(positions[0][:, 0])),
        fargs=(ax, positions, colors, title),
        repeat=loop,
        interval=50,
        blit=False,
    )

    if show:
        plt.show()

    if save_path:
        ani.save(
            save_path,
            writer="imagemagick",
            fps=100,
        )
