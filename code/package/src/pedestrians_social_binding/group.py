from pedestrians_social_binding.pedestrian import Pedestrian
from pedestrians_social_binding.trajectory_utils import *


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

    def get_members(self):
        return self.members

    def get_center_of_mass_trajectory(self):
        members_trajectories = [member.trajectory for member in self.members]
        simultaneous_taj = compute_simultaneous_observations(members_trajectories)

        simultaneous_time = simultaneous_taj[0][:, 0]
        x_members = np.stack([traj[:, 1] for traj in simultaneous_taj], axis=1)
        y_members = np.stack([traj[:, 2] for traj in simultaneous_taj], axis=1)
        z_members = np.stack([traj[:, 3] for traj in simultaneous_taj], axis=1)

        vx_members = np.stack([traj[:, 5] for traj in simultaneous_taj], axis=1)
        vy_members = np.stack([traj[:, 6] for traj in simultaneous_taj], axis=1)

        x_center_of_mass = np.sum(x_members, axis=1) / self.size
        y_center_of_mass = np.sum(y_members, axis=1) / self.size
        z_center_of_mass = np.sum(z_members, axis=1) / self.size

        vx_center_of_mass = np.sum(vx_members, axis=1) / self.size
        vx_center_of_mass = np.sum(vy_members, axis=1) / self.size

        v_center_of_mass = (vx_center_of_mass**2 + vx_center_of_mass**2) ** 0.5

        trajectory = np.stack(
            (
                simultaneous_time,
                x_center_of_mass,
                y_center_of_mass,
                z_center_of_mass,
                v_center_of_mass,
                vx_center_of_mass,
                v_center_of_mass,
            ),
            axis=1,
        )
        return trajectory
