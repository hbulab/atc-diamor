from __future__ import annotations
from asyncio import constants
from pedestrians_social_binding.utils import *
from pedestrians_social_binding.plot_utils import *
from pedestrians_social_binding.constants import *
from pedestrians_social_binding.trajectory_utils import *


class Pedestrian:
    """Class representing a pedestrian

    Attributes
    ----------
    ped_id : int, the pedestrian id
    trajectory : numpy array, the trajectory of the pedestrian
    groups : list of Group, the groups the pedestrian is part of
    env : Environment, the environment the pedestrian is in
    day : int, the day the pedestrian is in
    first_obs : int, the first observation of the pedestrian
    last_obs : int, the last observation of the pedestrian

    Constructor
    -----------
    ped_id : int, the pedestrian id
    env : Environment, the environment the pedestrian is in
    day : int, the day the pedestrian is in
    trajectory : numpy array, the trajectory of the pedestrian
    groups : list of Group, the groups the pedestrian is part of

    """

    def __init__(self, ped_id, env, day, trajectory, groups, is_non_group):
        self.ped_id = ped_id
        # trajectory is a numpy array with one line for each data point
        # and 7 columns :
        # time, x, y, z, v, vx, vy
        self.trajectory = trajectory
        self.groups = groups
        self.env = env
        self.day = day
        self.first_obs = trajectory[0, 0] if len(trajectory) else -1
        self.last_obs = trajectory[-1, 0] if len(trajectory) else -1
        self.is_non_group = is_non_group

    def __str__(self):
        """String representation of the pedestrian"""
        return f"Pedestrian({self.ped_id})"

    def __repr__(self):
        """String representation of the pedestrian"""
        return f"Pedestrian({self.ped_id})"

    def get_id(self) -> int:
        """Get the pedestrian id"""
        return self.ped_id

    def get_trajectory(self):
        """Get the pedestrian trajectory"""
        return self.trajectory

    def get_time(self):
        """Get the pedestrian time"""
        return self.trajectory[:, 0]

    def set_trajectory(self, trajectory):
        """Set the pedestrian trajectory"""
        self.trajectory = trajectory

    def get_first_obs(self):
        """Get the first observation of the pedestrian"""
        return self.first_obs

    def get_last_obs(self):
        """Get the last observation of the pedestrian"""
        return self.last_obs

    def get_trajectory_column(self, value):
        """Get the trajectory column corresponding to the value

        Parameters
        ----------
        value : str, the value to get the column for

        Returns
        -------
        numpy array, the trajectory column corresponding to the value
        """
        if value not in TRAJECTORY_COLUMNS:
            raise ValueError(
                f"Unknown threshold value {value}. Should be one of {list(TRAJECTORY_COLUMNS.keys())}"
            )
        return self.trajectory[:, TRAJECTORY_COLUMNS[value]]

    def get_position(self):
        """Get the pedestrian position"""
        return self.trajectory[:, 1:3]

    def shares_observations_with(self, pedestrian):
        """Check if the pedestrian shares observations with another pedestrian

        Parameters
        ----------
        pedestrian : Pedestrian, the pedestrian to check with

        Returns
        -------
        bool, True if the pedestrian shares observations with the other pedestrian, False otherwise
        """
        return not (
            self.last_obs < pedestrian.get_first_obs()
            or self.first_obs > pedestrian.get_last_obs()
        )

    def get_encountered_pedestrians(
        self,
        pedestrians,
        proximity_threshold: int | None = 4000,
        skip: list[int] = [],
        alone: bool | None = None,
    ) -> list[Pedestrian]:
        """Get the pedestrians encountered by the pedestrian

        Parameters
        ----------
        pedestrians : list of Pedestrian, the pedestrians to check with
        proximity_threshold : int, the proximity threshold to use
        skip : list of int, the pedestrians to skip
        alone : bool, whether to get alone or not alone encounters

        Returns
        -------
        list of Pedestrian, the encountered pedestrians
        """

        encounters = []
        for pedestrian in pedestrians:
            if (
                pedestrian.ped_id == self.ped_id or pedestrian.ped_id in skip
            ):  # don't compare with himself or ped to skip
                continue
            if not self.shares_observations_with(pedestrian):
                continue
            sim_traj, sim_traj_ped = compute_simultaneous_observations(
                [self.trajectory, pedestrian.get_trajectory()]
            )
            if len(sim_traj) == 0 or len(sim_traj_ped) == 0:
                continue

            if proximity_threshold is not None and (
                min(compute_interpersonal_distance(sim_traj, sim_traj_ped))
                > proximity_threshold
            ):
                continue
            encounters += [pedestrian]

        if alone is not None:
            if alone:
                encounters = compute_alone_encounters(
                    encounters, self, proximity_threshold
                )
            else:
                encounters = compute_not_alone_encounters(
                    encounters, self, proximity_threshold
                )
        return encounters

    def plot_2D_trajectory(
        self,
        ax=None,
        scale=True,
        animate=False,
        show=True,
        save_path=None,
        loop=False,
        gradient=False,
    ):
        """Plot the pedestrian trajectory

        Parameters
        ----------
        scale : bool, whether to scale the trajectory or not
        animate : bool, whether to animate the trajectory or not
        show : bool, whether to show the plot or not
        save_path : str, the path to save the plot to
        loop : bool, whether to loop the animation or not
        """

        if scale:
            boundaries = self.env.boundaries
        else:
            boundaries = None

        trajectory = self.trajectory

        if animate:
            plot_animated_2D_trajectory(
                trajectory,
                ax=ax,
                title=f"Trajectory for {self.ped_id}",
                boundaries=boundaries,
                show=show,
                save_path=save_path,
                loop=loop,
            )
        else:
            plot_static_2D_trajectory(
                trajectory,
                ax=ax,
                title=f"Trajectory for {self.ped_id}",
                boundaries=boundaries,
                show=show,
                save_path=save_path,
                gradient=gradient,
            )

    def get_undisturbed_trajectory(self, proximity_threshold, pedestrians, skip=[]):
        """Get the undisturbed trajectory of the pedestrian

        Parameters
        ----------
        proximity_threshold : int, the proximity threshold to use
        pedestrians : list of Pedestrian, the pedestrians to check with
        skip : list of int, the pedestrians to skip

        Returns
        -------
        numpy array, the undisturbed trajectory
        """
        traj = self.trajectory
        times_in_vicinity = np.array([])
        for pedestrian in pedestrians:
            if (
                pedestrian.ped_id == self.ped_id or pedestrian.ped_id in skip
            ):  # don't compare with members or ped to skip
                continue
            if not self.shares_observations_with(pedestrian):
                continue
            sim_traj_group, sim_traj_ped = compute_simultaneous_observations(
                [traj, pedestrian.get_trajectory()]
            )

            distance = compute_interpersonal_distance(sim_traj_group, sim_traj_ped)
            times_in_vicinity = np.union1d(
                times_in_vicinity, sim_traj_group[distance < proximity_threshold][:, 0]
            )

        return get_trajectory_not_at_times(traj, times_in_vicinity)
