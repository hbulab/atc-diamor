from pedestrians_social_binding.pedestrian import Pedestrian
from pedestrians_social_binding.trajectory_utils import *
from pedestrians_social_binding.plot_utils import *


class Group:
    """Class representing a group of pedestrians.
    A group is defined by a set of pedestrians, a group id, an environment, a day and a set of annotations.
    The annotations are a dictionary of the form {annotation_type: annotation_value}.

    ----------
    Constructor of the class.

    Parameters

    group_id: int, the id of the group
    members: list[Pedestrian], the list of pedestrians in the group
    env: Environment, the environment in which the group evolves
    day: str, the day of the observation
    annotations: dict, the annotations of the group

    """

    def __init__(self, group_id, members, env, day, annotations):
        self.group_id = group_id
        self.members = members
        self.size = len(members)
        self.env = env
        self.day = day
        self.annotations = annotations

        # We create a fake pedestrian corresponding to the group
        self.as_individual = Pedestrian(
            group_id,
            env,
            day,
            self.get_center_of_mass_trajectory(),
            [],
            is_non_group=None,
        )

    def __str__(self):
        """Returns a string representation of the group."""
        return f"Group({self.group_id})"

    def __repr__(self):
        """Returns a string representation of the group."""
        return f"Group({self.group_id})"

    def get_as_individual(self) -> Pedestrian:
        """Returns the individual pedestrian corresponding to the group."""
        return self.as_individual

    def get_id(self):
        """Returns the group id."""
        return self.group_id

    def get_size(self):
        """Returns the size of the group."""
        return self.size

    def get_members(self) -> list[Pedestrian]:
        """Returns the list of members of the group."""
        return self.members

    def get_annotation(self, annotation_type):
        """Returns the annotation of the group corresponding to the given annotation type."""
        return self.annotations.get(annotation_type, None)

    def get_interpersonal_distance(self):
        """Returns the interpersonal distance between the two pedestrians of the group."""

        if self.size != 2:
            raise ValueError(
                f"Cannot compute the interpersonal distance on a group with size {self.size}."
            )
        members_trajectories = [member.trajectory for member in self.members]
        traj_A, traj_B = compute_simultaneous_observations(members_trajectories)
        pos_A = traj_A[:, 1:3]
        pos_B = traj_B[:, 1:3]
        return compute_interpersonal_distance(pos_A, pos_B)

    def get_depth_and_breadth(self):
        """Returns the depth and breadth of the group.
        The depth is the distance between the two pedestrians of the group.
        The breadth is the distance between the center of mass of the group and the line defined by the two pedestrians.
        """
        if self.size != 2:
            raise ValueError(
                f"Cannot compute the breadth on a group with size {self.size}."
            )
        members_trajectories = [member.trajectory for member in self.members]
        traj_A, traj_B = compute_simultaneous_observations(members_trajectories)
        return compute_depth_and_breadth(traj_A, traj_B)

    def get_relative_orientation(self):
        """Returns the relative orientation of the group."""

        if self.size != 2:
            raise ValueError(
                f"Cannot compute the relative orientation on a group with size {self.size}."
            )
        members_trajectories = [member.trajectory for member in self.members]
        traj_A, traj_B = compute_simultaneous_observations(members_trajectories)
        return compute_relative_orientation(traj_A, traj_B)

    def get_absolute_difference_velocity(self):
        """Returns the absolute difference of velocity of the group."""

        if self.size != 2:
            raise ValueError(
                f"Cannot compute the absolute difference of velocity on a group with size {self.size}."
            )
        members_trajectories = [member.trajectory for member in self.members]
        traj_A, traj_B = compute_simultaneous_observations(members_trajectories)
        return compute_absolute_difference_velocity(traj_A, traj_B)

    def get_center_of_mass_trajectory(self):
        """Returns the center of mass trajectory of the group."""
        members_trajectories = [member.trajectory for member in self.members]
        return compute_center_of_mass(members_trajectories)

    def plot_2D_trajectory(
        self,
        scale=True,
        simultaneous=True,
        animate=False,
        show=True,
        save_path=None,
        loop=False,
    ):
        """Plots the 2D trajectory of the group.

        Parameters
        ----------
        scale: bool, whether to scale the trajectory to the environment boundaries
        simultaneous: bool, whether to plot the trajectories simultaneously or not
        animate: bool, whether to animate the trajectories or not
        show: bool, whether to show the plot or not
        save_path: str, the path where to save the plot
        loop: bool, whether to loop the animation or not
        """

        if scale:
            boundaries = self.env.boundaries
        else:
            boundaries = None

        trajectories = [m.get_trajectory() for m in self.members]
        ped_ids = [m.get_id() for m in self.members]

        if animate:
            plot_animated_2D_trajectories(
                trajectories,
                title=f"Trajectory for {self.group_id}",
                labels=ped_ids,
                simultaneous=simultaneous,
                boundaries=boundaries,
                show=show,
                save_path=save_path,
                loop=loop,
            )
        else:
            plot_static_2D_trajectories(
                trajectories,
                title=f"Trajectory for {self.group_id}",
                labels=ped_ids,
                simultaneous=simultaneous,
                boundaries=boundaries,
                show=show,
                save_path=save_path,
            )

    def get_encountered_pedestrians(self, proximity_threshold, pedestrians, skip=[]):
        """Returns the pedestrians encountered by the group.

        Parameters
        ----------
        proximity_threshold: float, the proximity threshold to consider an encounter
        pedestrians: list, the list of pedestrians to consider
        skip: list, the list of pedestrians to skip

        Returns
        -------
        list, the list of encountered pedestrians
        """

        encounters = []
        members_id = [m.get_id() for m in self.members]
        for pedestrian in pedestrians:
            if (
                pedestrian.ped_id in members_id or pedestrian.ped_id in skip
            ):  # don't compare with members or ped to skip
                continue
            members_trajectories = [m.get_trajectory() for m in self.members]
            if not have_simultaneous_observations(
                members_trajectories + [pedestrian.get_trajectory()]
            ):
                continue
            sim_trajs = compute_simultaneous_observations(
                members_trajectories + [pedestrian.get_trajectory()]
            )
            sim_traj_members = sim_trajs[:-1]
            sim_traj = sim_trajs[-1]
            min_distances = [
                min(compute_interpersonal_distance(traj, sim_traj))
                for traj in sim_traj_members
            ]
            if all(d > proximity_threshold for d in min_distances):
                continue
            encounters += [pedestrian]
        return encounters

    def get_undisturbed_trajectory(self, proximity_threshold, pedestrians, skip=[]):
        """Returns the undisturbed trajectory of the group.

        Parameters
        ----------
        proximity_threshold: float, the proximity threshold to consider an encounter
        pedestrians: list, the list of pedestrians to consider
        skip: list, the list of pedestrians to skip

        Returns
        -------
        array, the undisturbed trajectory
        """

        members_id = [m.get_id() for m in self.members]
        group_traj = self.get_center_of_mass_trajectory()
        times_in_vicinity = np.array([])
        for pedestrian in pedestrians:
            if (
                pedestrian.ped_id in members_id or pedestrian.ped_id in skip
            ):  # don't compare with members or ped to skip
                continue
            if not have_simultaneous_observations(
                [group_traj, pedestrian.get_trajectory()]
            ):
                continue
            sim_traj_group, sim_traj_ped = compute_simultaneous_observations(
                [group_traj, pedestrian.get_trajectory()]
            )

            distance = compute_interpersonal_distance(sim_traj_group, sim_traj_ped)
            times_in_vicinity = np.union1d(
                times_in_vicinity, sim_traj_group[distance < proximity_threshold][:, 0]
            )

        return get_trajectory_not_at_times(group_traj, times_in_vicinity)
        # if all(d > proximity_threshold for d in min_distances):
        #     continue
        # encounters += [pedestrian]
        # return undisturbed_trajectory

    # def get_undisturbed_trajectory(self, proximity_threshold, pedestrians, skip=[]):
    #     encounters = []
    #     members_id = [m.get_id() for m in self.members]
    #     undisturbed_trajectories = [m.get_trajectory() for m in self.members]
    #     for pedestrian in pedestrians:
    #         if (
    #             pedestrian.ped_id in members_id or pedestrian.ped_id in skip
    #         ):  # don't compare with members or ped to skip
    #             continue
    #         sim_trajs = compute_simultaneous_observations(
    #             undisturbed_trajectories + [pedestrian.get_trajectory()]
    #         )
    #         if len(sim_trajs[0]) == 0:
    #             continue
    #         sim_traj_members = sim_trajs[:-1]
    #         sim_traj = sim_trajs[-1]
    #         distances_to_members = [
    #             compute_interpersonal_distance(traj, sim_traj)
    #             for traj in sim_traj_members
    #         ]
    #         undisturbed_trajectories = [
    #             compute_simultaneous_observations(
    #                 [undisturbed_trajectory, traj[d > proximity_threshold]]
    #             )[0]
    #             for undisturbed_trajectory, d, traj in zip(
    #                 undisturbed_trajectories, distances_to_members, sim_trajs
    #             )
    #         ]
    #         # if all(d > proximity_threshold for d in min_distances):
    #         #     continue
    #         # encounters += [pedestrian]
    #     return undisturbed_trajectories
