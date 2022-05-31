import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_static_2D_trajectory(x, y, ped_id, boundaries, show=True, save_path=None):
    plt.scatter(x / 1000, y / 1000, c="cornflowerblue", s=10)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([boundaries["xmin"] / 1000, boundaries["xmax"] / 1000])
    plt.ylim([boundaries["ymin"] / 1000, boundaries["ymax"] / 1000])
    plt.title(f"Trajectory of pedestrian {ped_id}")
    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)


def plot_animated_2D_trajectory(
    x, y, ped_id, boundaries, show=True, save_path=None, loop=False
):

    colors = ["cornflowerblue"] * len(x)

    fig, ax = plt.subplots()
    ax.scatter([], [], c=[])  # plot of x and y in time

    def animate(i, ax, pos_Nx, pos_Ny, colorsN):
        ax.clear()
        ax.scatter(pos_Nx[:i], pos_Ny[:i], c=colorsN[:i], s=2)
        ax.set_title(f"Trajectory of {ped_id}")
        ax.axis(
            [
                boundaries["xmin"] / 1000,
                boundaries["xmax"] / 1000,
                boundaries["ymin"] / 1000,
                boundaries["ymax"] / 1000,
            ]
        )

    ax.axis(
        [
            boundaries["xmin"] / 1000,
            boundaries["xmax"] / 1000,
            boundaries["ymin"] / 1000,
            boundaries["ymax"] / 1000,
        ]
    )
    ax.set_title(f"Trajectory of {ped_id}")
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
            fps=20,
        )
