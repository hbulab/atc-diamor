import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pedestrians_social_binding.trajectory_utils import *
from pedestrians_social_binding.constants import *


def plot_static_2D_trajectory(
    trajectory: np.ndarray,
    title: str = None,
    boundaries: dict = None,
    vel: bool = False,
    show: bool = True,
    save_path: str = None,
    ax: matplotlib.axes.Axes = None,
):
    """Plot the trajectory of a pedestrian, as an image

    Parameters
    ----------
    pedestrian : np.ndarray. A pedestrian
    title: str, A title for the animation, by default None
    boundaries : dict, optional. The boundaries of the environment to be use as axis limits, by default None
    show : bool, optional. Whether or not the image should be displayed, by default True
    save_path : str, optional. The path to the file where the image will be saved, by default None
    ax : matplotlib.axes.Axes, optional, by default None
    """

    if not ax:
        fig, ax = plt.subplots()

    x, y = trajectory[:, 1] / 1000, trajectory[:, 2] / 1000
    ax.scatter(x, y, c="cornflowerblue", s=10)

    if vel:
        vx, vy = trajectory[:, 5] / 1000, trajectory[:, 6] / 1000
        ax.quiver(x, y, vx, vy)

    if boundaries:
        ax.set_xlim([boundaries["xmin"] / 1000, boundaries["xmax"] / 1000])
        ax.set_ylim([boundaries["ymin"] / 1000, boundaries["ymax"] / 1000])

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)


def plot_animated_2D_trajectory(
    trajectory: np.ndarray,
    title: str = None,
    vel: bool = False,
    colors: list[str] = None,
    boundaries: dict = None,
    show: bool = True,
    save_path: str = None,
    loop: bool = False,
    background: str = None,
):
    """Plot the trajectory of a pedestrian, as an animation

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    title : str, optional
        A title for the animation, by default None
    vel : bool, optional
        Whether or not the velocities should be displayed as arrows, by default False
    boundaries : dict, optional
        The boundaries of the environment to be use as axis limits, by default None
    show : bool, optional
        Whether or not the animation should be displayed, by default True
    save_path : str, optional
        The path to the file where the animation will be saved, by default None
    loop : bool, optional
        Whether or not the animation should loop, by default False
    background : str, optional
        Path to an image to use as background, by default None
    """
    position = trajectory[:, 1:3] / 1000
    velocity = trajectory[:, 5:7] / 1000
    # x, y = pedestrian.get_trajectory_column("x"), pedestrian.get_trajectory_column("y")

    if colors is None:
        colors = ["cornflowerblue"] * len(position)

    fig, ax = plt.subplots()
    ax.scatter([], [], c=[])  # plot of x and y in time

    if boundaries:
        xmin, xmax = boundaries["xmin"] / 1000, boundaries["xmax"] / 1000
        ymin, ymax = boundaries["ymin"] / 1000, boundaries["ymax"] / 1000
    else:
        xmin, xmax = min(position[:, 0]), max(position[:, 0])
        ymin, ymax = min(position[:, 1]), max(position[:, 1])

    def animate(i, ax, position, velocity, colors):
        ax.clear()
        ax.scatter(position[:i, 0], position[:i, 1], c=colors[:i], s=10)
        if vel:
            ax.arrow(
                position[i, 0],
                position[i, 1],
                velocity[i, 0],
                velocity[i, 1],
                color="black",
                head_length=1,
                head_width=0.5,
            )
        ax.set_title(title)
        ax.axis([xmin, xmax, ymin, ymax])
        if background is not None:
            img = plt.imread(background)
            ax.imshow(img, extent=[xmin, xmax, ymin, ymax])

    ax.axis([xmin, xmax, ymin, ymax])

    if background is not None:
        img = plt.imread(background)
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax])

    if title is not None:
        ax.set_title(title)
    ax.set_autoscale_on(False)

    ani = animation.FuncAnimation(
        fig,
        animate,
        range(len(position)),
        fargs=(ax, position, velocity, colors),
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
    trajectories: list[np.ndarray],
    labels: list[str] = None,
    simultaneous: bool = False,
    show_direction: bool = False,
    boundaries: dict = None,
    colors: list[str] = None,
    title: str = None,
    show: bool = True,
    save_path: str = None,
    ax: matplotlib.axes.Axes = None,
):
    """Plot the trajectories of a set of pedestrians

    Parameters
    ----------
    trajectories : list[np.ndarray]
        A list of trajectories
    labels : list[str], optional
        A list of labels, by default None
    simultaneous : bool, optional
        Whether or not the trajectories should be cropped to the simultaneous observations, by default False
    show_direction: bool, optional
        Whether or not an arrow showing the direction should be plot,
        based on the velocities, by default False
    boundaries : dict, optional
        The boundaries of the environment to be used as axis limits, by default None
    colors : list[str], optional
        A list of colors to be used, by default None
    title : str, optional
        The title of the figure, by default None
    show : bool, optional
        Whether or not the image should be displayed, by default True
    save_path : str, optional
         The path to the file where the image will be saved, by default None
    ax : matplotlib.axes.Axes, optional
        The axes on which the trajectories should be plotted, by default None
    
    """
    n_traj = len(trajectories)

    if not ax:
        fig, ax = plt.subplots()

    if simultaneous:
        trajectories = compute_simultaneous_observations(trajectories)

    if not colors:
        if n_traj > len(COLORS):
            colors = np.random.choice(COLORS, n_traj)
        else:
            colors = COLORS[:n_traj]

    if labels is None:
        zip_labels = n_traj * [None]  # no labels
    else:
        zip_labels = labels

    for label, trajectory, color in zip(zip_labels, trajectories, colors):
        x, y = trajectory[:, 1], trajectory[:, 2]
        alphas = np.linspace(0, 1, len(x))
        ax.scatter(x / 1000, y / 1000, c=color, alpha=alphas, s=10, label=label)

        if show_direction:
            middle = len(trajectory) // 2
            ax.arrow(
                x[middle] / 1000,
                y[middle] / 1000,
                trajectory[middle, 5] / 1000,
                trajectory[middle, 6] / 1000,
                color="black",
                head_length=1,
                head_width=0.5,
            )

    ax.axis("scaled")

    if boundaries:
        plt.xlim([boundaries["xmin"] / 1000, boundaries["xmax"] / 1000])
        plt.ylim([boundaries["ymin"] / 1000, boundaries["ymax"] / 1000])

    if title is not None:
        plt.title(title)
    ax.set_xlabel("x (m)")

    if labels is not None:
        plt.legend()

    ax.set_ylabel("y (m)")
    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)


def plot_animated_2D_trajectories(
    trajectories: list[np.ndarray],
    labels: list[str] = None,
    simultaneous: bool = False,
    vel: bool = False,
    boundaries: dict = None,
    vicinity: float = None,
    colors: list[str] = None,
    title: str = None,
    show: bool = True,
    save_path: str = None,
    loop: bool = False,
):
    """Plot the trajectory of a set of pedestrians, as an animation

    Parameters
    ----------
    trajectories : list[np.ndarray]
        A list of trajectories
    labels : list[str], optional
         A list of labels, by default None
    simultaneous : bool, optional
        Whether or not the trajectories should be cropped to the simultaneous observations, by default False
    vel : bool, optional
        Whether or not the velocities should be displayed as arrows, by default False
    boundaries : dict, optional
        The boundaries of the environment to be use as axis limits, by default None
    vicinity : float, optional
        A vicinity to be drawn, by default None
    colors : list[str], optional
        A list of colors to use for the trajectories, by default None
    title : str, optional
        A title for the animation, by default None
    show : bool, optional
        Whether or not the animation should be displayed, by default True
    save_path : str, optional
        The path to the file where the animation will be saved, by default None
    loop : bool, optional
        Whether or not the animation should loop, by default False
    """
    n_traj = len(trajectories)

    if simultaneous:
        trajectories = compute_simultaneous_observations(trajectories)

    trajectories = get_padded_trajectories(trajectories)

    positions = [traj[:, 1:3] / 1000 for traj in trajectories]
    velocities = [traj[:, 5:7] / 1000 for traj in trajectories]

    if not colors:
        if n_traj > len(COLORS):
            colors = np.random.choice(COLORS, n_traj)
        else:
            colors = COLORS[:n_traj]

    fig, ax = plt.subplots()

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

    def animate(i, ax, labels, positions, velocities, vicinity, colors, title):
        ax.clear()
        ax.axis([xmin, xmax, ymin, ymax])

        if labels is None:
            zip_labels = n_traj * [None]  # no labels
        else:
            zip_labels = labels

        ax.set_aspect("equal", "box")
        if vicinity:  # set the size of the marker for the vicinity
            M = ax.transData.get_matrix()
            scale = M[0, 0]
            s = s = (scale * vicinity / 1000) ** 2
            for position, color in zip(positions, colors):
                ax.scatter(
                    position[i, 0],
                    position[i, 1],
                    c=color,
                    s=s,
                    alpha=0.3,
                )

        if vel:
            for position, velocity in zip(positions, velocities):
                ax.arrow(
                    position[i, 0],
                    position[i, 1],
                    velocity[i, 0],
                    velocity[i, 1],
                    color="black",
                    head_length=1,
                    head_width=0.5,
                )

        for label, position, color in zip(zip_labels, positions, colors):
            ax.scatter(position[:i, 0], position[:i, 1], c=color, s=10, label=label)
        if title is not None:
            ax.set_title(title)
        if labels is not None:
            ax.legend()

    ani = animation.FuncAnimation(
        fig,
        animate,
        range(len(positions[0][:, 0])),
        fargs=(ax, labels, positions, velocities, vicinity, colors, title),
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

def plot_baseline(trajectory: np.ndarray, 
                  max_dev , 
                  soc_binding: str, 
                  group: bool, 
                  id: int = None, 
                  boundaries: dict = None,
                    colors: list[str] = None,
                    n_average: int = 4,
                    ax = None,
                    fig = None,
                    show: bool = True,
):

    """ This function plots the baseline trajectory of a pedestrian.

    Parameters
    ----------
    trajectory : np.array
        The trajectory of the pedestrian.
    max_dev : dict
        A dictionary containing the maximum lateral deviation of the pedestrian.
        The dictionary has the following structure:
            - max_dev["max lateral deviation"] = max_deviation
            - max_dev["position of max lateral deviation"] = [time, x, y]  
            - max_dev["start_vel"] = [x, y]
    soc_binding : str
        The social binding of the pedestrian.
    group : bool
        True if the pedestrian is in a group, False otherwise.
    id : int
        The id of the pedestrian or the group.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    point_of_max_deviation = max_dev["position of max lateral deviation"]
    start_vel = np.nanmean(trajectory[:n_average, 5:7], axis=0)
    x_start_plot = trajectory[0,1]
    y_start_plot = trajectory[0,2]
    x_end_plot = start_vel[0] * 1000 + x_start_plot
    y_end_plot = start_vel[1] * 1000 + y_start_plot

    vel_perpandicular = np.array([start_vel[1], -start_vel[0]])
    x_start_perp_plot = point_of_max_deviation[1]
    y_start_perp_plot = point_of_max_deviation[2]
    x_end_perp_plot = vel_perpandicular[0] * 1000 + x_start_perp_plot
    y_end_perp_plot = vel_perpandicular[1] * 1000 + y_start_perp_plot
    x_second_end_perp_plot = -vel_perpandicular[0] * 1000 + x_start_perp_plot
    y_second_end_perp_plot = -vel_perpandicular[1] * 1000 + y_start_perp_plot


    # plot the trajectory
    if (group):
        if(soc_binding == "other") :
            color = "pink"
        else:
            color = colors[soc_binding]
    else:
        color = "blue"
    ax.set_xlim([boundaries["xmin"] / 1000, boundaries["xmax"] / 1000])
    ax.set_ylim([boundaries["ymin"] / 1000, boundaries["ymax"] / 1000])
    ax.scatter(trajectory[:,1] / 1000,trajectory[:,2] / 1000, s=10, c=color)
    ax.scatter(point_of_max_deviation[1] / 1000, point_of_max_deviation[2] / 1000, s=10, c="black")
    ax.set_aspect("equal", "box")
    ax.plot([x_start_plot / 1000, x_end_plot / 1000], [y_start_plot / 1000, y_end_plot / 1000], c="purple", label="velocity vector")
    ax.set_xlabel('X Coord', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coord', fontsize=12, fontweight='bold')
    if(group):
        ax.set_title('Plot of the baseline for group number : ' + str(id))
    else:
        ax.set_title('Plot of the baseline for non group pedestrian ' + str(id))
    ax.plot([x_end_perp_plot / 1000, x_second_end_perp_plot / 1000], [y_end_perp_plot / 1000, y_second_end_perp_plot/1000]
                , c="green", label="perpendicular of the velocity vector")

    if(show) :
        plt.show()
    else :
        return fig,ax
