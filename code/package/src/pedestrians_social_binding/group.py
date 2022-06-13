from pedestrians_social_binding.pedestrian import Pedestrian
from pedestrians_social_binding.trajectory_utils import *
from pedestrians_social_binding.plot_utils import *


class Group:
    def __init__(self, group_id, members, env, day, annotations):

        self.group_id = group_id
        self.members = members
        self.size = len(members)
        self.env = env
        self.day = day
        self.annotations = annotations

        self.as_individual = Pedestrian(
            group_id, env, day, self.get_center_of_mass_trajectory(), []
        )

    def __str__(self):
        return f"Group({self.group_id})"

    def __repr__(self):
        return f"Group({self.group_id})"

    def get_as_individual(self) -> Pedestrian:
        return self.as_individual

    def get_id(self):
        return self.group_id

    def get_size(self):
        return self.size

    def get_members(self) -> list[Pedestrian]:
        return self.members

    def get_annotation(self, annotation_type):
        return self.annotations.get(annotation_type, None)

    def get_interpersonal_distance(self):

        if self.size != 2:
            raise ValueError(
                f"Cannot compute the interpersonal distance on a group with size {self.size}."
            )
        members_trajectories = [member.trajectory for member in self.members]
        traj_A, traj_B = compute_simultaneous_observations(members_trajectories)
        return compute_interpersonal_distance(traj_A, traj_B)

    def get_relative_orientation(self):

        if self.size != 2:
            raise ValueError(
                f"Cannot compute the interpersonal distance on a group with size {self.size}."
            )
        members_trajectories = [member.trajectory for member in self.members]
        traj_A, traj_B = compute_simultaneous_observations(members_trajectories)
        traj_G = self.get_center_of_mass_trajectory()
        return compute_relative_orienation(traj_G, traj_A, traj_B)

    def get_center_of_mass_trajectory(self):
        members_trajectories = [member.trajectory for member in self.members]
        simultaneous_traj = compute_simultaneous_observations(members_trajectories)

        simultaneous_time = simultaneous_traj[0][:, 0]
        x_members = np.stack([traj[:, 1] for traj in simultaneous_traj], axis=1)
        y_members = np.stack([traj[:, 2] for traj in simultaneous_traj], axis=1)
        z_members = np.stack([traj[:, 3] for traj in simultaneous_traj], axis=1)

        vx_members = np.stack([traj[:, 5] for traj in simultaneous_traj], axis=1)
        vy_members = np.stack([traj[:, 6] for traj in simultaneous_traj], axis=1)

        x_center_of_mass = np.sum(x_members, axis=1) / self.size
        y_center_of_mass = np.sum(y_members, axis=1) / self.size
        z_center_of_mass = np.sum(z_members, axis=1) / self.size

        vx_center_of_mass = np.sum(vx_members, axis=1) / self.size
        vy_center_of_mass = np.sum(vy_members, axis=1) / self.size

        v_center_of_mass = (vx_center_of_mass**2 + vx_center_of_mass**2) ** 0.5

        trajectory = np.stack(
            (
                simultaneous_time,
                x_center_of_mass,
                y_center_of_mass,
                z_center_of_mass,
                v_center_of_mass,
                vx_center_of_mass,
                vy_center_of_mass,
            ),
            axis=1,
        )
        return trajectory

    def plot_2D_trajectory(
        self,
        scale=True,
        simultaneous=True,
        animate=False,
        show=True,
        save_path=None,
        loop=False,
    ):

        if scale:
            boundaries = self.env.boundaries
        else:
            boundaries = None

        if animate:
            plot_animated_2D_trajectories(
                self.members,
                simultaneous=simultaneous,
                boundaries=boundaries,
                show=show,
                save_path=save_path,
                loop=loop,
            )
        else:
            plot_static_2D_trajectories(
                self.members,
                simultaneous=simultaneous,
                boundaries=boundaries,
                show=show,
                save_path=save_path,
            )

    def get_encountered_pedestrians(self, proximity_threshold, pedestrians, skip=[]):
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
