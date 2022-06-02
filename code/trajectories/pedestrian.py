from asyncio import constants
from trajectories.utils import *
from trajectories.plot_utils import *
from trajectories.constants import *


class Pedestrian:
    def __init__(self, ped_id, env, day, trajectory, groups):
        self.ped_id = ped_id
        # trajectory is a numpy array with one line for each data point
        # and 7 columns :
        # time, x, y, z, v, vx, vy
        self.trajectory = trajectory
        self.groups = groups
        self.env = env
        self.day = day

    def __str__(self):
        return f"Pedestrian({self.ped_id})"

    def __repr__(self):
        return f"Pedestrian({self.ped_id})"

    def get_trajectory(self):
        return self.trajectory

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def get_trajectory_column(self, value):
        if value not in TRAJECTORY_COLUMNS:
            raise ValueError(
                f"Unknown threshold value {value}. Should be one of {list(TRAJECTORY_COLUMNS.keys())}"
            )
        return self.trajectory[:, TRAJECTORY_COLUMNS[value]]

    def get_position(self):
        return self.trajectory[1:3]

    def get_encounters(self, proximity_threshold, pedestrians):
        encounters = []
        for pedestrian in pedestrians:
            if pedestrian.ped_id == self.ped_id:
                continue
            if not have_simultaneous_observations(
                [self.trajectory, pedestrian.get_trajectory()]
            ):
                continue
            if (
                min(
                    compute_interpersonal_distance(
                        self.trajectory, pedestrian.get_trajectory()
                    )
                )
                > proximity_threshold
            ):
                continue
            encounters += [pedestrian]
        return encounters

    def plot_2D_trajectory(
        self,
        scale=False,
        animate=False,
        colors=None,
        show=True,
        save_path=None,
        loop=False,
    ):

        if scale:
            boundaries = self.env.boundaries
        else:
            boundaries = None

        if animate:
            plot_animated_2D_trajectory(self, boundaries, show, save_path, loop)
        else:
            plot_static_2D_trajectory(self, boundaries, show, save_path)
