import matplotlib.pyplot as plt
import matplotlib.animation as animation

from trajectories.trajectory_utils import *
from trajectories.constants import *


def plot_static_2D_trajectory(pedestrian, boundaries=None, show=True, save_path=None):
    x, y = pedestrian.get_trajectory()[:, 1], pedestrian.get_trajectory()[:, 2]
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
    x, y = pedestrian.get_trajectory()[:, 1], pedestrian.get_trajectory()[:, 2]

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
