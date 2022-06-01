from trajectories.environment import Environment
from trajectories.threshold import Threshold

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == "__main__":

    atc = Environment("atc", data_dir="../data/formatted")

    XMIN, XMAX = 5000, 48000
    YMIN, YMAX = -27000, 8000

    GRID_SIZE = 100
    n_bin_x = int(np.ceil((XMAX - XMIN) / GRID_SIZE) + 1)
    n_bin_y = int(np.ceil((YMAX - YMIN) / GRID_SIZE) + 1)
    grid = np.zeros((n_bin_x, n_bin_y))

    # add thresholds to keep only corridor data
    threshold_x = Threshold("x", XMIN, XMAX)
    threshold_y = Threshold("y", YMIN, YMAX)

    # get the occupancy grid for all pedestrians in the corridor
    pedestrians = atc.get_pedestrians(thresholds=[threshold_x, threshold_y])

    for pedestrian in pedestrians:
        x = pedestrian.get_trajectory_column("x")
        y = pedestrian.get_trajectory_column("y")

        nx = np.ceil((x - XMIN) / GRID_SIZE).astype("int")
        ny = np.ceil((y - YMIN) / GRID_SIZE).astype("int")

        grid[nx, ny] += 1

    max_val = np.max(grid)
    grid /= max_val

    xi = np.linspace(0, (XMAX - XMIN) / 1000, n_bin_x)
    yi = np.linspace(0, (YMAX - YMIN) / 1000, n_bin_y)

    plt.figure()
    cmesh = plt.pcolormesh(xi, yi, grid.T, cmap="inferno_r", shading="auto")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    plt.colorbar(cmesh, cax=cax)
    axes.set_aspect("equal")
    plt.show()

    # get the occupancy grid for just the dyads
    dyads = atc.get_groups(size=2, thresholds=[threshold_x, threshold_y])

    for dyad in dyads:
        x = dyad.get_as_pedestrian().get_trajectory_column("x")
        y = dyad.get_as_pedestrian().get_trajectory_column("y")

        nx = np.ceil((x - XMIN) / GRID_SIZE).astype("int")
        ny = np.ceil((y - YMIN) / GRID_SIZE).astype("int")

        grid[nx, ny] += 1

    max_val = np.max(grid)
    grid /= max_val

    xi = np.linspace(0, (XMAX - XMIN) / 1000, n_bin_x)
    yi = np.linspace(0, (YMAX - YMIN) / 1000, n_bin_y)

    plt.figure()
    cmesh = plt.pcolormesh(xi, yi, grid.T, cmap="inferno_r", shading="auto")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    plt.colorbar(cmesh, cax=cax)
    axes.set_aspect("equal")
    plt.show()
