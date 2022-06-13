from asyncio import constants
from pedestrians_social_binding.utils import *
from pedestrians_social_binding.plot_utils import *
from pedestrians_social_binding.constants import *


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

    def get_id(self):
        return self.ped_id

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
        return self.trajectory[:, 1:3]

    def get_encountered_pedestrians(
        self, pedestrians, proximity_threshold=4000, skip=[]
    ):
        encounters = []
        for pedestrian in pedestrians:
            if (
                pedestrian.ped_id == self.ped_id or pedestrian.ped_id in skip
            ):  # don't compare with himself or ped to skip
                continue
            if not have_simultaneous_observations(
                [self.trajectory, pedestrian.get_trajectory()]
            ):
                continue
            sim_traj, sim_traj_ped = compute_simultaneous_observations(
                [self.trajectory, pedestrian.get_trajectory()]
            )
            if proximity_threshold is not None and (
                min(compute_interpersonal_distance(sim_traj, sim_traj_ped))
                > proximity_threshold
            ):
                continue
            encounters += [pedestrian]
        return encounters

    def plot_2D_trajectory(
        self,
        scale=True,
        animate=False,
        show=True,
        save_path=None,
        loop=False,
    ):

        if scale:
            boundaries = self.env.boundaries
        else:
            boundaries = None

        trajectory = self.trajectory

        if animate:
            plot_animated_2D_trajectory(
                trajectory,
                title=f"Trajectory for {self.ped_id}",
                boundaries=boundaries,
                show=show,
                save_path=save_path,
                loop=loop,
            )
        else:
            plot_static_2D_trajectory(
                trajectory,
                title=f"Trajectory for {self.ped_id}",
                boundaries=boundaries,
                show=show,
                save_path=save_path,
            )

    def get_undisturbed_trajectory(self, proximity_threshold, pedestrians, skip=[]):
        traj = self.trajectory
        times_in_vicinity = np.array([])
        for pedestrian in pedestrians:
            if (
                pedestrian.ped_id == self.ped_id or pedestrian.ped_id in skip
            ):  # don't compare with members or ped to skip
                continue
            if not have_simultaneous_observations([traj, pedestrian.get_trajectory()]):
                continue
            sim_traj_group, sim_traj_ped = compute_simultaneous_observations(
                [traj, pedestrian.get_trajectory()]
            )

            distance = compute_interpersonal_distance(sim_traj_group, sim_traj_ped)
            times_in_vicinity = np.union1d(
                times_in_vicinity, sim_traj_group[distance < proximity_threshold][:, 0]
            )

        return get_trajectory_not_at_times(traj, times_in_vicinity)
