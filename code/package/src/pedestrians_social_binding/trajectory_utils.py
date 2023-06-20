from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from pedestrians_social_binding.group import Group
    from pedestrians_social_binding.pedestrian import Pedestrian
    from pedestrians_social_binding.threshold import Threshold

from pedestrians_social_binding.constants import *
from pedestrians_social_binding.parameters import *

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from typing import List


cross = lambda x, y, axis=None: np.cross(x, y, axis=axis)  # annotation bug


def compute_simultaneous_observations(trajectories: list[np.ndarray]) -> list:
    """Find the section of the trajectories that correspond to simultaneous observations

    Parameters
    ----------
    trajectories : list
        List of trajectories

    Returns
    -------
    list
        The list of trajectories with simultaneous observations (i.e. same time stamps)
    """

    simult_time = trajectories[0][:, 0]

    for trajectory in trajectories[1:]:
        simult_time = np.intersect1d(simult_time, trajectory[:, 0])

    simult_trajectories = []
    for trajectory in trajectories:
        time_mask = np.isin(trajectory[:, 0], simult_time)
        simult_trajectory = trajectory[time_mask, :]
        simult_trajectories += [simult_trajectory]

    return simult_trajectories


def have_simultaneous_observations(trajectories: list[np.ndarray]) -> bool:
    """Check if a list of trajectories have some observations at same time stamps

    Parameters
    ----------
    trajectories : list
        The list of trajectories

    Returns
    -------
    bool
        True if the list of trajectories have at least one identical time stamps, False otherwise
    """
    return len(compute_simultaneous_observations(trajectories)[0]) > 0


def get_trajectory_at_times(trajectory: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Get the observation from the trajectory, at the given times

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    times : np.ndarray
        An array of time stamps

    Returns
    -------
    np.ndarray
        Observations from the trajectory at the given time stamps
    """
    trajectory_at_times = np.full((len(times), 7), np.nan)
    times_traj = trajectory[:, 0]
    times_in_times_traj = np.isin(times, times_traj)
    times_traj_in_times = np.isin(times_traj, times)

    
    trajectory_at_times[times_in_times_traj] = trajectory[times_traj_in_times]

    return trajectory_at_times


def get_trajectory_in_time_frame(
    trajectory: np.ndarray, t_min: int = None, t_max: int = None
) -> np.ndarray:
    """Get the observation from the trajectory, at the given times

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    t_min : int
        the minimum time, by default None
    t_max : int
        the maximum time, by default None

    Returns
    -------
    np.ndarray
        Observations from the trajectory in between the given time boundaries
    """
    if t_min is not None:
        trajectory = trajectory[trajectory[:, 0] >= t_min]
    if t_max is not None:
        trajectory = trajectory[trajectory[:, 0] <= t_max]
    return trajectory


def get_trajectories_at_times(
    trajectories: list[np.ndarray], times: np.ndarray
) -> list[np.ndarray]:
    """Get the observations at the given times for all trajectories

    Parameters
    ----------
    trajectories : list[np.ndarray]
        A list of trajectories
    times : np.ndarray
        An array of time stamps

    Returns
    -------
    list[np.ndarray]
        The list of trajectories with only the observations at the given time stamps
    """
    trajectories_at_time = []
    for trajectory in trajectories:
        trajectories_at_time += [get_trajectory_at_times(trajectory, times)]

    return trajectories_at_time


def get_trajectory_not_at_times(
    trajectory: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Returns observations from the trajectory that are not at the given time stamps

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    times : np.ndarray
        An array of time stamps

    Returns
    -------
    np.ndarray
        All observations from the trajectory that are at the given time stamps
    """
    times_traj = trajectory[:, 0]
    at_times = np.isin(times_traj, times)
    return trajectory[np.logical_not(at_times)]


def get_trajectories_not_at_times(
    trajectories: list[np.ndarray], times: np.ndarray
) -> list[np.ndarray]:
    """Get the observations not at the given times for all trajectories

    Parameters
    ----------
    trajectories : list[np.ndarray]
        A list of trajectories
    times : np.ndarray
        An array of time stamps

    Returns
    -------
    list[np.ndarray]
        The list of trajectories with only the observations not at the given time stamps
    """
    trajectories_not_at_time = []
    for trajectory in trajectories:
        trajectories_not_at_time += [get_trajectory_not_at_times(trajectory, times)]

    return trajectories_not_at_time


def get_padded_trajectories(trajectories: list[np.ndarray]) -> list[np.ndarray]:
    """Get trajectories padded with NaN values, so that each trajectory has
    observations for timestamps in all trajectories

    Parameters
    ----------
    trajectories : list[np.ndarray]
        A list of trajectories

    Returns
    -------
    list[np.ndarray]
        List of padded trajectories
    """
    all_times = trajectories[0][:, 0]
    for trajectory in trajectories[1:]:
        all_times = np.union1d(all_times, trajectory[:, 0])
    padded_trajectories = []
    for trajectory in trajectories:
        indices_times = np.isin(all_times, trajectory[:, 0])
        padded_trajectory = np.full((len(all_times), 7), np.nan)
        padded_trajectory[indices_times, :] = trajectory
        padded_trajectories += [padded_trajectory]

    return padded_trajectories


def compute_interpersonal_distance(
    trajectory_A: np.ndarray, trajectory_B: np.ndarray
) -> np.ndarray:
    """Compute the pair-wise distances between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory

    Returns
    -------
    np.ndarray
        1D array containing the pair-wise distances
    """

    sim_traj_A, sim_traj_B = compute_simultaneous_observations(
        [trajectory_A, trajectory_B]
    )
    pos_A = sim_traj_A[:, 1:3]
    pos_B = sim_traj_B[:, 1:3]

    d = np.linalg.norm(pos_A - pos_B, axis=1)

    return d


def compute_absolute_difference_velocity(
    trajectory_A: np.ndarray, trajectory_B: np.ndarray
) -> np.ndarray:
    """Compute the absolute difference of the velocity of two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory

    Returns
    -------
    np.ndarray
        1D array containing the pair-wise distances
    """

    sim_traj_A, sim_traj_B = compute_simultaneous_observations(
        [trajectory_A, trajectory_B]
    )
    vel_A = sim_traj_A[:, 5:7]
    vel_B = sim_traj_B[:, 5:7]

    w = np.abs(np.linalg.norm(vel_A - vel_B, axis=1))

    return w


def compute_relative_direction(
    trajectory_A: np.ndarray, trajectory_B: np.ndarray
) -> str:
    """Compute the relative direction between two trajectories. First, compute
    the instantaneous relative direction at all time stamps based on the dot product between
    the velocities. Then, aggregates the results if a direction is predominant.

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory

    Returns
    -------
    str
        The value of the relative direction ("cross", "opposite", "same")
    """
    sim_traj_A, sim_traj_B = compute_simultaneous_observations(
        [trajectory_A, trajectory_B]
    )
    if len(sim_traj_A) < 1:
        return None

    pos_A = sim_traj_A[:, 1:3]
    pos_B = sim_traj_B[:, 1:3]
    v_A = sim_traj_A[:, 5:7]
    v_B = sim_traj_B[:, 5:7]

    # d_AB is the vector from A to B
    # d_AB = pos_B - pos_A

    # dot product of vA and vB
    v_d_dot = np.sum(v_A * v_B, axis=1)

    norm_product = np.linalg.norm(v_A, axis=1) * np.linalg.norm(v_B, axis=1)
    norm_product[norm_product == 0] = np.nan

    cos_vA_vB = v_d_dot / norm_product

    # rel_pos = np.zeros(cos_vA_vB.shape)

    n_val = len(cos_vA_vB)

    n_same = np.sum(cos_vA_vB > REL_DIR_ANGLE)
    n_opposite = np.sum(cos_vA_vB < -REL_DIR_ANGLE)
    n_cross = n_val - n_same - n_opposite

    if n_same > REL_DIR_MIN_PERC * n_val:
        return "same"
    elif n_opposite > REL_DIR_MIN_PERC * n_val:
        return "opposite"
    elif n_cross > REL_DIR_MIN_PERC * n_val:
        return "cross"
    else:
        return None


def filter_pedestrian(pedestrian: Pedestrian, threshold: Threshold) -> Pedestrian:
    """Filter the trajectory of a pedestrian based on a threshold

    Parameters
    ----------
    pedestrian : Pedestrian
        A pedestrian object
    threshold : Threshold
        A threshold object

    Returns
    -------
    Pedestrian
        The pedestrian with thresholded trajectory. If the pedestrians needs to be discarded,
        None is returned.
    """
    value = threshold.get_value()
    min_val = threshold.get_min_value()
    max_val = threshold.get_max_value()

    if value == "d":  # threshold on the distance
        position = pedestrian.get_position()
        d = np.linalg.norm(position[-1] - position[0])
        if (
            (
                min_val is not None
                and max_val is not None
                and d <= max_val
                and d >= min_val
            )
            or (min_val is not None and d >= min_val)
            or (max_val is not None and d <= max_val)
        ):
            return pedestrian
        else:
            return None

    elif value == "n":  # threshold on the number of observation
        trajectory = pedestrian.get_trajectory()
        if min_val is not None and max_val is not None:
            keep = len(trajectory) > min_val and len(trajectory) < max_val
        elif min_val is not None:
            keep = len(trajectory) > min_val
        else:
            keep = len(trajectory) < max_val
        if keep:
            return pedestrian
        else:
            return None

    elif value == "t":  # threshold on the time
        time = pedestrian.get_column("t")
        t_obs = time[-1] - time[0]
        if (
            (
                min_val is not None
                and max_val is not None
                and t_obs <= max_val
                and t_obs >= min_val
            )
            or (min_val is not None and t_obs >= min_val)
            or (max_val is not None and t_obs <= max_val)
        ):
            return pedestrian
        else:
            return None

    # elif value == "theta":  # threshold on turning angles
    #     position = pedestrian.get_position()
    #     turning_angles = np.abs(compute_turning_angles(position))
    #     if min_val is not None and max_val is not None:
    #         bad_points_indices = (
    #             np.where((turning_angles < min_val) | (turning_angles > max_val))[0] + 1
    #         )
    #     elif min_val is not None:
    #         bad_points_indices = np.where(turning_angles < min_val)[0] + 1
    #     else:
    #         bad_points_indices = np.where(turning_angles > max_val)[0] + 1
    #     # cut around the points
    #     for i in range(-5, 6):
    #         bad_points_indices = np.union1d(bad_points_indices, bad_points_indices + i)
    #     bad_points_indices = bad_points_indices[bad_points_indices < len(position)]
    #     bad_points_indices = bad_points_indices[bad_points_indices > 0]

    #     threshold_indices = np.setdiff1d(range(len(position)), bad_points_indices)
    #     # print(threshold_indices)

    # elif value == "v":
    elif value in ["x", "y"]:  # threshold on the position
        column = pedestrian.get_trajectory_column(value)
        mean_value = np.nanmean(column)
        if min_val is not None and max_val is not None:
            threshold_indices = np.where((column >= min_val) & (column <= max_val))[0]
        elif min_val is not None:
            threshold_indices = np.where(column >= min_val)[0]
        else:
            threshold_indices = np.where(column <= max_val)[0]
        if len(threshold_indices) > 0:
            trajectory = pedestrian.get_trajectory()[threshold_indices, :]
            pedestrian.set_trajectory(trajectory)
            return pedestrian
    else:
        column = pedestrian.get_trajectory_column(value)
        mean_value = np.nanmean(column)
        if min_val is not None and max_val is not None:
            # threshold_indices = np.where((column >= min_val) & (column <= max_val))[0]
            keep = mean_value >= min_val and mean_value <= max_val
        elif min_val is not None:
            # threshold_indices = np.where(column >= min_val)[0]
            keep = mean_value >= min_val
        else:
            # threshold_indices = np.where(column <= max_val)[0]
            keep = mean_value <= max_val
        if keep:
            return pedestrian
        else:
            return None
    # else:
    #     column = pedestrian.get_trajectory_column(value)
    #     mean_value = np.nanmean(column)
    #     if min_val is not None and max_val is not None:
    #         threshold_indices = np.where((column >= min_val) & (column <= max_val))[0]
    #     elif min_val is not None:
    #         threshold_indices = np.where(column >= min_val)[0]
    #     else:
    #         threshold_indices = np.where(column <= max_val)[0]
    #     if len(threshold_indices) > 0:
    #         trajectory = pedestrian.get_trajectory()[threshold_indices, :]
    #         pedestrian.set_trajectory(trajectory)
    #         return pedestrian
    #     else:
    #         return None


def filter_pedestrians(
    pedestrians: list[Pedestrian], threshold: Threshold
) -> list[Pedestrian]:
    """Filters pedestrians given a threshold.

    Parameters
    ----------
    pedestrians : list[Pedestrian]
        A list of pedestrians objects
    threshold : Threshold
        A threshold object

    Returns
    -------
    list[Pedestrian]
        A list of pedestrians whose trajectories have been thresholded. Some pedestrians
        might be discarded and the length of the returned list will be smaller than the length
        of the input pedestrians.
    """
    filtered_pedestrians = []
    for pedestrian in pedestrians:
        pedestrian = filter_pedestrian(pedestrian, threshold)
        if pedestrian:
            filtered_pedestrians += [pedestrian]

    return filtered_pedestrians


def filter_group(group: Group, threshold: Threshold) -> bool:
    """Decides if a group needs to be discarded based on a threshold

    Parameters
    ----------
    group : Group
        A group object
    threshold : Threshold
        A threshold object

    Returns
    -------
    bool
        True if the group satisfies the threshold, False otherwise.
    """
    value = threshold.get_value()
    min_val = threshold.get_min_value()
    max_val = threshold.get_max_value()

    if value == "delta":
        d_AB = group.get_interpersonal_distance()

        if min_val is not None and max_val is not None:
            condition = np.logical_and(d_AB >= min_val, d_AB <= max_val)
        elif min_val is not None:
            condition = d_AB >= min_val
        else:
            condition = d_AB <= max_val
        return np.all(condition)


def translate_position(position: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Translate a position by a given 2D vector

    Parameters
    ----------
    position : np.ndarray
        A position
    translation : np.ndarray
        A 2D translation vector

    Returns
    -------
    np.ndarray
        The translated position
    """
    return position + translation


def rotate_position(position: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a position by a given angle, around the origin (0, 0)

    Parameters
    ----------
    position : np.ndarray
        A position
    angle : float
        An angle

    Returns
    -------
    np.ndarray
        The rotated position
    """
    r_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    r_position = np.dot(r_mat, position.T).T
    return r_position


def compute_interpersonal_distance(pos_A: np.ndarray, pos_B: np.ndarray) -> np.ndarray:
    """Compute the distance between two position, at each time stamp. Assumes that the
    values of position are for corresponding time stamps.

    Parameters
    ----------
    pos_A : np.ndarray
        A position
    pos_B : np.ndarray
        A position

    Returns
    -------
    np.ndarray
        The array of distance at all time stamps
    """
    dist_AB = np.linalg.norm(pos_A - pos_B, axis=1)
    return dist_AB


def compute_depth_and_breadth(traj_A: np.ndarray, traj_B: np.ndarray) -> np.ndarray:
    """Compute the distance between two position, at each time stamp, along the
    direction of the group's motion (the depth) and along
    direction orthogonal to the group's motion (the breadth). Assumes that the
    values of position are for corresponding time stamps.

    Parameters
    ----------
    traj_A : np.ndarray
        A trajectory
    traj_B : np.ndarray
        A trajectory

    Returns
    -------
    np.ndarray
        The array of breadth at all time stamps
    """
    group_velocity = compute_center_of_mass([traj_A, traj_B])[:, 5:7]
    u_v = (
        group_velocity / np.linalg.norm(group_velocity, axis=1)[:, None]
    )  # unitary vector
    pos_A = traj_A[:, 1:3]
    pos_B = traj_B[:, 1:3]
    vec_AB = pos_B - pos_A
    # the depth is component of vec_AB along u_v and the breadth is the component
    # of vec_AB along the direction orthogonal to u_v
    vec_depth = np.sum(vec_AB * u_v, axis=1)[:, None] * u_v
    depth = np.linalg.norm(vec_depth, axis=1)
    vec_breadth = vec_AB - vec_depth
    breadth = np.linalg.norm(vec_breadth, axis=1)

    dist_AB = np.linalg.norm(pos_A - pos_B, axis=1)
    # print(
    #     dist_AB[0],
    #     depth[0],
    #     breadth[0],
    #     dist_AB[0] - (depth[0] ** 2 + breadth[0] ** 2) ** 0.5,
    # )

    return depth, breadth


def compute_center_of_mass(trajectories: list[np.ndarray]) -> np.ndarray:
    """Computes the center of mass of a list of trajectories. Position and velocities are
    the average of all trajectories.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        A list of trajectories

    Returns
    -------
    np.ndarray
        The trajectory of the center of mass
    """
    simultaneous_traj = compute_simultaneous_observations(trajectories)
    n_traj = len(trajectories)

    simultaneous_time = simultaneous_traj[0][:, 0]
    x_members = np.stack([traj[:, 1] for traj in simultaneous_traj], axis=1)
    y_members = np.stack([traj[:, 2] for traj in simultaneous_traj], axis=1)
    z_members = np.stack([traj[:, 3] for traj in simultaneous_traj], axis=1)

    vx_members = np.stack([traj[:, 5] for traj in simultaneous_traj], axis=1)
    vy_members = np.stack([traj[:, 6] for traj in simultaneous_traj], axis=1)

    x_center_of_mass = np.sum(x_members, axis=1) / n_traj
    y_center_of_mass = np.sum(y_members, axis=1) / n_traj
    z_center_of_mass = np.sum(z_members, axis=1) / n_traj

    vx_center_of_mass = np.sum(vx_members, axis=1) / n_traj
    vy_center_of_mass = np.sum(vy_members, axis=1) / n_traj

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


def compute_relative_orientation(traj_A: np.ndarray, traj_B: np.ndarray) -> np.ndarray:
    """Computes the relative orientation of members of a group given trajectories of the members.

    Parameters
    ----------
    traj_A : np.ndarray
        Trajectory of one member of the group
    traj_B : np.ndarray
        Trajectory of ont member of the group

    Returns
    -------
    np.ndarray
        The array of relative orientation (angles) of the group.
    """
    traj_center_of_mass = compute_center_of_mass([traj_A, traj_B])
    [traj_center_of_mass, traj_A, traj_B] = compute_simultaneous_observations(
        [traj_center_of_mass, traj_A, traj_B]
    )
    v_G = traj_center_of_mass[:, 5:7]
    pos_A = traj_A[:, 1:3]
    pos_B = traj_B[:, 1:3]
    d_AB = pos_B - pos_A
    rel_orientation_AB = np.arctan2(d_AB[:, 1], d_AB[:, 0]) - np.arctan2(
        v_G[:, 1], v_G[:, 0]
    )
    rel_orientation_AB[rel_orientation_AB > np.pi] -= 2 * np.pi
    rel_orientation_AB[rel_orientation_AB < -np.pi] += 2 * np.pi

    d_BA = pos_A - pos_B
    rel_orientation_BA = np.arctan2(d_BA[:, 1], d_BA[:, 0]) - np.arctan2(
        v_G[:, 1], v_G[:, 0]
    )
    rel_orientation_BA[rel_orientation_BA > np.pi] -= 2 * np.pi
    rel_orientation_BA[rel_orientation_BA < -np.pi] += 2 * np.pi

    return np.concatenate((rel_orientation_AB, rel_orientation_BA))


def compute_continuous_sub_trajectories(
    trajectory: np.ndarray, max_gap: int = 2000
) -> list[np.ndarray]:
    """Breaks down a trajectory in to a list of sub-trajectories that have maximum time
    gaps of max_gap

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    max_gap : int, optional
        The maximum temporal gap allowed in a trajectory, by default 2000

    Returns
    -------
    list[np.ndarray]
        The list of continuous sub-trajectories (i.e. with no gap larger than max_gap)
    """

    not_nan_indices = np.where(np.logical_not(np.isnan(trajectory[:, 1])))[0]
    trajectory_not_nan = trajectory[not_nan_indices, :]
    t = trajectory_not_nan[:, 0]
    delta_t = t[1:] - t[:-1]

    jumps_indices = np.where(delta_t > max_gap)[0]

    sub_trajectories = []
    s = 0
    for j in jumps_indices:
        sub_trajectories += [trajectory_not_nan[s : j + 1, :]]
        s = j + 1
    if s < len(t):
        sub_trajectories += [trajectory_not_nan[s:, :]]

    return sub_trajectories

def compute_continuous_sub_trajectories_using_time(trajectory: np.ndarray, max_gap: int = 2000
) -> list[np.ndarray]:
    """Breaks down a trajectory in to a list of sub-trajectories that have maximum time
    gaps of max_gap

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    max_gap : int, optional
        The maximum temporal gap allowed in a trajectory, by default 2000

    Returns
    -------
    list[np.ndarray]
        The list of continuous sub-trajectories (i.e. with no gap larger than max_gap)
    """

    t = trajectory[:, 0]
    delta_t = t[1:] - t[:-1]

    jumps_indices = np.where(delta_t > max_gap)[0]

    sub_trajectories = []
    s = 0
    for j in jumps_indices:
        sub_trajectories += [trajectory[s : j + 1, :]]
        s = j + 1
    if s < len(t):
        sub_trajectories += [trajectory[s:, :]]

    return sub_trajectories

def compute_continuous_sub_trajectories_using_distance(trajectory: np.ndarray, max_distance: int = 5000, min_length: int=5) -> List[List[np.ndarray], List[float]]:
    """Breaks down a trajectory in to a list of sub-trajectories that have maximum time
    gaps of max_gap

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    max_gap : int, optional
        The maximum temporal gap allowed in a trajectory, by default 2000

    Returns
    -------
    list[np.ndarray]
        The list of continuous sub-trajectories (i.e. with no gap larger than max_gap)
    """

    s = 0

    sub_sub_trajectories = []
    break_trigger = False
    liste_of_length = []


    for s in range(len(trajectory)):
        for i in range(s+1,len(trajectory)):
            break_trigger = False
            delta = trajectory[i, 1:3] - trajectory[s, 1:3]
            for diff in delta:
                distance = np.sqrt(np.sum(delta**2))
                if distance > max_distance:
                    if i - s >= min_length:
                        sub_sub_trajectories += [trajectory[s:i, :]]
                        liste_of_length += [distance]
                        break_trigger = True
                        break
            if break_trigger:
                break

    if(len(sub_sub_trajectories) == 0):
        return None

    return sub_sub_trajectories, liste_of_length




def compute_maximum_lateral_deviation(
    position: np.ndarray, scaled: bool = True
) -> float:
    """Computes the maximum lateral deviation over the trajectory (the maximum distance from
    points of the trajectories to the line joining the first and last point of the trajectory).

    Parameters
    ----------
    position : np.ndarray
        A position
    scaled : bool, optional
        Whether or not the value is scaled by the distance between the first
        and last point of the trajectory, by default True

    Returns
    -------
    float
        The value for the maximum lateral deviation (or for the scaled maximum lateral deviation)
    """

    start_point = position[0]
    end_point = position[-1]
    middle_points = position[1:-1]
    # for all points except first and last, compute the distance between the line
    # from start S to end E and the point P
    # i.e. (SE x PE) / ||PE||
    distances_to_straight_line = np.abs(
        cross(end_point - start_point, middle_points - start_point)
    ) / np.linalg.norm(end_point - start_point)
    if scaled:  # divide by the distance from P to E
        distances_to_straight_line /= np.linalg.norm(end_point - start_point)

    max_distance = np.max(distances_to_straight_line)

    return max_distance


def compute_maximum_lateral_deviation_using_vel(
    traj: np.ndarray,
    n_average=3,
    interpolate: bool = False,
) -> float:
    """Computes the maximum lateral deviation over the trajectory (the maximum distance from points of the trajectories to the line joining the first and last point of the trajectory).

    Parameters
    ----------
    traj : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to average the velocity over, by default 3
    interpolate : bool, optional
        Whether or not to interpolate the velocity, by default False

    Returns
    -------
    float
        The value for the maximum lateral deviation
    """
    pos = traj[:, 1:3]
    vel = traj[:, 5:7]
    delta_ts = (traj[1:, 0] - traj[:-1, 0]) / 1000

    start_point = pos[0]
    middle_points = pos[1:-1]

    start_vel = np.nanmean(traj[:n_average, 5:7], axis=0)

    if not interpolate:
        distances_to_straight_line = np.abs(
            cross(start_vel, middle_points - start_point)
        ) / np.linalg.norm(start_vel)

        max_distance = np.max(distances_to_straight_line)
        return max_distance

    else:
        # position after half time step, extrapolating velocity
        extr_pos = pos[:-1] + delta_ts[:, None] / 2 * vel[:-1]
        points = np.concatenate((middle_points, extr_pos))
        distances_to_straight_line = np.abs(
            cross(start_vel, points - start_point)
        ) / np.linalg.norm(start_vel)

        max_distance = np.max(distances_to_straight_line)
        return max_distance


def compute_maximum_lateral_deviation_using_vel_2(
    traj: np.ndarray,
    n_average=3,
    interpolate: bool = False,
    length: float = None,
) -> dict["max_lateral_deviation": float, "position of max lateral deviation": np.ndarray, "start_vel": np.ndarray, "length_of_trajectory": float]:
    """Computes the maximum lateral deviation over the trajectory (the maximum distance from points of the trajectories to the line joining the first and last point of the trajectory).

    Parameters
    ----------
    traj : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to average the velocity over, by default 3
    interpolate : bool, optional
        Whether or not to interpolate the velocity, by default False

    Returns
    -------
    dict["max_lateral_deviation": float, "position of max lateral deviation": np.ndarray]
        The value for the maximum lateral deviation and the position of the point where it occurs
    """
    dict_return = {"max_lateral_deviation": 0, "position of max lateral deviation": np.array([0, 0, 0, 0, 0, 0 ,0]), "start_vel": np.array([0, 0]), "length_of_trajectory": 0}

    pos = traj[:, 1:3]
    vel = traj[:, 5:7]
    delta_ts = (traj[1:, 0] - traj[:-1, 0]) / 1000

    start_point = pos[0]
    middle_points = pos[1:]

    start_vel = np.nanmean(traj[:n_average, 5:7], axis=0)

    if not interpolate:
        distances_to_straight_line = np.abs(
            cross(start_vel, middle_points - start_point)
        ) / np.linalg.norm(start_vel)

        max_distance = np.max(distances_to_straight_line)
        dict_return["max_lateral_deviation"] = max_distance
        dict_return["position of max lateral deviation"] = traj[np.argmax(distances_to_straight_line)+1, :]
        dict_return["start_vel"] = start_vel
        dict_return["length_of_trajectory"] = length
        return dict_return

    else:
        # position after half time step, extrapolating velocity
        extr_pos = pos[:-1] + delta_ts[:, None] / 2 * vel[:-1]
        points = np.concatenate((middle_points, extr_pos))
        distances_to_straight_line = np.abs(
            cross(start_vel, points - start_point)
        ) / np.linalg.norm(start_vel)

        max_distance = np.max(distances_to_straight_line)
        dict_return["max_lateral_deviation"] = max_distance
        dict_return["position of max lateral deviation"] = traj[np.argmax(distances_to_straight_line), :]
        dict_return["start_vel"] = start_vel
        dict_return["length_of_trajectory"] = length
        return dict_return


def compute_net_displacement(position: np.ndarray) -> float:
    """Computes the net displacement of the trajectory (the
    distance between the first and last point)

    Parameters
    ----------
    position : np.ndarray
        A position

    Returns
    -------
    float
        The value for the net displacement
    """
    start_point = position[0]
    end_point = position[-1]
    net_displacement = np.linalg.norm(end_point - start_point)
    return net_displacement


def compute_gross_displacement(position: np.ndarray) -> float:
    """Computes the gross displacement (the
    sum of the distance between each consecutive points of the trajectory)

    Parameters
    ----------
    position : np.ndarray
        A position

    Returns
    -------
    float
        The value for the gross displacement
    """
    gross_displacement = np.sum(np.linalg.norm(position[:-1] - position[1:], axis=1))
    return gross_displacement


def compute_straightness_index(position: np.ndarray) -> float:
    """Computes the straightness index of a trajectory. The straightness index
    is defined as the D/L where D is the net displacement of the trajectory (the
    distance between the first and last point) and L is the gross displacement (the
    sum of the distance between each consecutive points of the trajectory)

    Parameters
    ----------
    position : np.ndarray
        A position

    Returns
    -------
    float
        The value for the straightness index
    """
    net_displacement = compute_net_displacement(position)
    gross_displacement = compute_gross_displacement(position)
    return net_displacement / gross_displacement


def compute_turning_angles(position: np.ndarray) -> np.ndarray:
    """Computes the turning angles of a trajectory

    Parameters
    ----------
    position : np.ndarray
        A position

    Returns
    -------
    np.ndarray
        The array of turning angles along the trajectory
    """
    step_vectors = position[1:, :] - position[:-1, :]
    turning_angles = np.arctan2(step_vectors[1:, 0], step_vectors[1:, 1]) - np.arctan2(
        step_vectors[:-1, 0], step_vectors[:-1, 1]
    )
    turning_angles[turning_angles > np.pi] -= 2 * np.pi
    turning_angles[turning_angles < -np.pi] += 2 * np.pi

    return turning_angles


def compute_angles(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Compute the angles between two arrays of 2D vectors

    Parameters
    ----------
    vectors_1 : np.ndarray
        Array of 2D vectors
    vectors_2 : np.ndarray
        Array of 2D vectors

    Returns
    -------
    np.ndarray
        The array of angles
    """
    angles = np.arctan2(vectors_1[:, 0], vectors_1[:, 1]) - np.arctan2(
        vectors_2[:, 0], vectors_2[:, 1]
    )
    angles[angles > np.pi] -= 2 * np.pi
    angles[angles < -np.pi] += 2 * np.pi
    return angles


def rediscretize_position(position: np.ndarray) -> np.ndarray:
    """Transforms the trajectory so that the distance between each point is fixed

    Parameters
    ----------
    position : np.ndarray
        A position

    Returns
    -------
    np.ndarray
        The position with a constant step size
    """
    step_sizes = np.linalg.norm(position[:-1] - position[1:], axis=1)
    n_points = len(position)
    q = np.min(step_sizes)
    current_goal_index = 1
    current_point = position[0]
    current_goal = position[current_goal_index]
    rediscretized_position = [current_point]
    # print(q)
    while True:
        d_point_goal = np.linalg.norm(current_goal - current_point)
        if d_point_goal > q:  # there is space until the next trajectory point
            l = q / d_point_goal
            current_point = l * current_goal + (1 - l) * current_point
        else:  # no more space, the points needs to be on the next segment
            # at the intersection between the line from current_goal to next_goal
            # and the circle centered on current_point with radius q
            if current_goal_index == n_points - 1:
                break
            next_goal_index = current_goal_index + 1
            next_goal = position[next_goal_index]
            u = (next_goal - current_goal) / np.linalg.norm(next_goal - current_goal)
            delta = 4 * np.dot(u, current_goal - current_point) ** 2 - (
                4 * (np.linalg.norm(current_goal - current_point) ** 2 - q**2)
            )
            m1 = (-2 * np.dot(u, current_goal - current_point) - delta**0.5) / 2
            m2 = (-2 * np.dot(u, current_goal - current_point) + delta**0.5) / 2
            m = max(m1, m2)
            next_point = current_goal + m * u
            current_point = next_point
            current_goal = next_goal
            current_goal_index += 1

        rediscretized_position += [current_point]
    return np.array(rediscretized_position)


def compute_sinuosity(position: np.ndarray) -> float:
    """Computes the sinuosity of the trajectory. Sinuosity is defined as 1.18 * s/q where
    s is the standard deviation of the turning angles of the trajectory and q is the step size
    of the trajectory

    Parameters
    ----------
    position : np.ndarray
        A positon

    Returns
    -------
    float
        The value for the sinuosity
    """
    rediscretized_position = rediscretize_position(position)
    step_size = np.linalg.norm(rediscretized_position[1] - rediscretized_position[0])
    turning_angles = compute_turning_angles(rediscretized_position)
    sinuosity = 1.18 * np.std(turning_angles) / step_size**0.5
    return sinuosity


def compute_area_under_the_curve(position: np.ndarray, scaled: bool = False) -> float:
    """Computes the area under the curve of the trajectory

    Parameters
    ----------
    position : np.ndarray
        A position
    scaled : bool, optional
        If True, the area is scaled by the gross displacement of the trajectory, by default False

    Returns
    -------
    float
        The area under the curve
    """
    start_point = position[0]
    end_point = position[-1]
    # for all points, compute the distance between the line
    # from start S to end E and the point P
    # i.e. (SE x PE) / ||PE||
    distances_to_straight_line = np.abs(
        cross(end_point - start_point, position - start_point)
    ) / np.linalg.norm(end_point - start_point)
    # compte the integral using the trapezoid
    # compute the projection of the trajectory points onto the straight line
    # get the bases of the trapezoid
    distances_to_first_point = distance.cdist([start_point], position)[0]
    cumul_bases = (
        distances_to_first_point**2 - distances_to_straight_line**2
    ) ** 0.5
    bases = cumul_bases[1:] - cumul_bases[:-1]

    # areas to the right are negative
    projections = start_point + cumul_bases[:, None] * (
        (end_point - start_point) / np.linalg.norm(end_point - start_point)
    )
    # plt.scatter(position[:, 0], position[:, 1])
    # plt.scatter(projections[:, 0], projections[:, 1])
    # plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]])
    # plt.axis("equal")
    # plt.show()
    vec_val = position - projections
    sign = np.sign(np.sum(vec_val * projections, axis=1))
    distances_to_straight_line *= sign

    areas = (
        bases * (distances_to_straight_line[:-1] + distances_to_straight_line[1:]) / 2
    )

    area_under_the_curve = np.sum(areas)
    if scaled:  # divide by the distance from P to E squared
        area_under_the_curve /= np.linalg.norm(end_point - start_point) ** 2
    return area_under_the_curve


def compute_deflection(
    position: np.ndarray, measure: str = "straightness_index"
) -> float:
    """Computes the deflection on the trajectory given a deflection measure

    Parameters
    ----------
    position : np.ndarray
        A position
    measure : str, optional
        The deflection measure to be user, by default "straightness_index", one of "straightness_index",
        "maximum_scaled_lateral_deviation", "maximum_lateral_deviation", "sinuosity", "area_under_curve"

    Returns
    -------
    float
        The deflection value
    """
    if measure == "straightness_index":
        return compute_straightness_index(position)
    elif measure == "maximum_scaled_lateral_deviation":
        return compute_maximum_lateral_deviation(position, scaled=True)
    elif measure == "maximum_lateral_deviation":
        return compute_maximum_lateral_deviation(position, scaled=False)
    elif measure == "sinuosity":
        return compute_sinuosity(position)
    elif measure == "area_under_curve":
        return compute_area_under_the_curve(position)
    elif measure == "scaled_area_under_curve":
        return compute_area_under_the_curve(position, scaled=True)
    else:
        raise ValueError(
            f"Unknown deflection measure {measure}. Expected one of {ALL_DEFLECTION_MEASURES}."
        )


def get_pieces(
    position: np.ndarray, piece_size: int, overlap: bool = False, delta: int = 100
) -> list[np.ndarray]:
    """Breaks up a trajectory in to pieces of given length

    Parameters
    ----------
    position : np.ndarray
        A position
    piece_size : int
        The length of the pieces
    overlap : bool, optional
        Whether or not overlapping pieces should be returned, by default False
    delta : int, optional
        The maximum difference allowed between the length of the pieces
        requested and the pieces returned, by default 100

    Returns
    -------
    list[np.ndarray]
        The list of pieces
    """
    start = position[0]
    end = position[1]

    points_distances = np.triu(distance.cdist(position, position, "euclidean"), k=0)

    indices_smaller = np.argwhere(np.abs(points_distances - piece_size) < delta)

    pieces = []
    current_min = 0
    for start, end in indices_smaller:
        # check if there is some overlaping, and skip piece
        if not overlap and start < current_min:
            continue
        current_min = end

        # get the trajectory between these points
        pieces += [position[start : end + 1, :]]
    return pieces


def get_random_pieces(position: np.ndarray, n_pieces=20) -> list[np.ndarray]:
    """Extract random pieces from a trajectory

    Parameters
    ----------
    position : np.ndarray
        A position
    n_pieces : int
        Number of pieces for which the deflection will be computed, by default 20
    Returns
    -------
    list[np.ndarray]
        The list of pieces
    """
    n_points = len(position)
    pieces = []
    while len(pieces) < n_pieces:
        i1 = np.random.randint(0, n_points - 1)
        i2 = np.random.randint(0, n_points - 1)
        min_i = min(i1, i2)
        max_i = max(i1, i2)
        if len(position[min_i:max_i, :]) <= 2:
            continue
        pieces += [position[min_i:max_i, :]]
    return pieces


def get_random_pieces_trajectory(
    trajectory: np.ndarray, n_pieces=20
) -> list[np.ndarray]:
    """Extract random pieces from a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_pieces : int
        Number of pieces for which the deflection will be computed, by default 20
    Returns
    -------
    list[np.ndarray]
        The list of pieces
    """
    n_points = len(trajectory)
    pieces = []
    while len(pieces) < n_pieces:
        i1 = np.random.randint(0, n_points - 1)
        i2 = np.random.randint(0, n_points - 1)
        if i1 == i2:
            continue
        min_i = min(i1, i2)
        max_i = max(i1, i2)
        pieces += [trajectory[min_i:max_i, :]]
    return pieces


def get_random_pieces_normal_distribution(
    position: np.ndarray, n_pieces=20, mu=6, sigma=0.6
) -> list[np.ndarray]:
    """Extract random pieces from a trajectory

    Parameters
    ----------
    position : np.ndarray
        A position
    n_pieces : int
        Number of pieces for which the deflection will be computed, by default 20
    Returns
    -------
    list[np.ndarray]
        The list of pieces
    """
    n_points = len(position)
    pieces = []
    while len(pieces) < n_pieces:
        len_traj = int(round(np.random.normal(mu, sigma)))
        if len_traj > n_points:
            continue
        max_start = n_points - len_traj
        start = np.random.randint(0, max_start + 1)
        pieces += [position[start : start + len_traj, :]]
    return pieces


def compute_piecewise_deflections(
    position: np.ndarray,
    piece_size: int,
    overlap: bool = False,
    delta: int = 100,
    measure: str = "straightness_index",
) -> list[float]:
    """Compute the deflection using the given method on all pieces of the given length.

    Parameters
    ----------
    position : np.ndarray
        A position
    piece_size : int
        The length of the pieces
    overlap : bool, optional
        Whether or not overlapping pieces should be returned, by default False
    delta : int, optional
        The maximum difference allowed between the length of the pieces
        requested and the pieces returned, by default 100
    measure : str, optional
        The deflection measure to be used, by default "straightness_index"

    Returns
    -------
    list[float]
        The deflection values for all pieces
    """
    pieces = get_pieces(position, piece_size=piece_size, overlap=overlap, delta=delta)
    deflections = [
        compute_deflection(piece, measure=measure)
        for piece in pieces
        if len(piece) >= 3
    ]
    return deflections


def compute_deflections_random_pieces(
    position: np.ndarray,
    n_pieces=20,
    measure: str = "straightness_index",
) -> list[float]:
    """Compute the deflection using the given method on random pieces

    Parameters
    ----------
    position : np.ndarray
        A position
    n_pieces : int
        Number of pieces for which the deflection will be computed, by default 20
    measure : str, optional
        The deflection measure to be used, by default "straightness_index"

    Returns
    -------
    list[float]
        The deflection values for all pieces
    """
    pieces = get_random_pieces(position, n_pieces)
    deflections = [
        compute_deflection(piece, measure=measure)
        for piece in pieces
        if len(piece) >= 3
    ]
    pieces_sizes = [
        compute_net_displacement(piece) for piece in pieces if len(piece) > 3
    ]
    return deflections, pieces_sizes


def get_time_step(trajectory: np.ndarray) -> float:
    """Computes the time step of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory

    Returns
    -------
    float
        The most frequent time step in the trajectory
    """

    if len(trajectory) <= 1:
        raise ValueError(
            f"Trajectory should have at least two data points ({len(trajectory)} found)."
        )
    times = trajectory[:, 0]
    # find most frequent time step
    traj_time_steps = times[1:] - times[:-1]
    values, counts = np.unique(traj_time_steps, return_counts=True)
    ind = np.argmax(counts)
    step = values[ind]
    return step


def resample_trajectory(trajectory: np.ndarray, sampling_time: int = 500) -> np.ndarray:
    """Resample the trajectory using the given time step

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    sampling_time : int, optional
        The time step to use (in ms), by default 500

    Returns
    -------
    np.ndarray
        The trajectory sampled a intervals of given time
    """
    current_time_step = get_time_step(trajectory)
    if (
        current_time_step < 1.2 * sampling_time
        and current_time_step > 0.8 * sampling_time
    ):  # nothing to do
        return trajectory

    times = trajectory[:, 0]
    traj_time_steps = times[1:] - times[:-1]

    rows_to_keep = [0]
    current_step = 0
    for i, time_step in enumerate(traj_time_steps):
        if current_step >= sampling_time:
            rows_to_keep += [i]
            current_step = 0
        current_step += time_step
    return trajectory[rows_to_keep, ...]


def compute_time_to_collision(obs_1: np.ndarray, obs_2: np.ndarray, delta=500) -> float:
    """Compute the expected time to collision between two pedestrians, given their current position
    and instantaneous velocity. Start by computing the collision point between the straight line trajectories
    and verify that both pedestrians reach this location at the same time (with a certain margin delta in ms)

    Parameters
    ----------
    obs_1 : np.ndarray
        An observation point (position, velocity)
    obs_2 : np.ndarray
        An observation point (position, velocity)
    delta : int, optional
        The margin to account for a real collision (i.e. pedestrians reach the collision at the same
        time more or less delta, |t1-t2| < delta), by default 500

    Returns
    -------
    float
        _description_
    """
    pos_1 = obs_1[1:3]
    vel_1 = obs_1[5:7]
    pos_2 = obs_2[1:3]
    vel_2 = obs_2[5:7]

    # find intersection point of line from pos_1 directed by vel_1 to
    # line from pos_2 directed by vel_2
    # i.e solve the system : P = pos_1 + t * vel_1 = pos_2 + t * vel_2
    # solve with Cramer's rule
    m1 = np.array([[pos_2[0] - pos_1[0], -vel_2[0]], [pos_2[1] - pos_1[1], -vel_2[1]]])
    m2 = np.array([[vel_1[0], pos_2[0] - pos_1[0]], [vel_1[1], pos_2[1] - pos_1[1]]])
    m3 = np.array([[vel_1[0], -vel_2[0]], [vel_1[1], -vel_2[1]]])
    t_1_coll = np.linalg.det(m1) / np.linalg.det(m3)
    t_2_coll = np.linalg.det(m2) / np.linalg.det(m3)

    if t_1_coll < 0 or t_2_coll < 0 or 1000 * abs(t_1_coll - t_2_coll) > delta:
        return -1, None
    else:
        t_coll = (t_1_coll + t_2_coll) / 2
        pos_coll = pos_1 + t_coll * vel_1
        return t_coll, pos_coll


def align_trajectories_at_origin(
    trajectory_ref: np.ndarray, trajectories: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Transform the trajectories in such a way that position for trajectory_ref will always
     be at the origin and velocity for A will be aligned along the positive x axis.
     Other trajectories are moved to the same reference frame

    Parameters
    ----------
    trajectory_ref : np.ndarray
        The trajectory that will be aligned with the x axis
    trajectories : list[np.ndarray]
        A list of trajectories to transform in that reference frame

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The aligned trajectories
    """

    # build the rotation matrix
    rotation_matrices = np.zeros(
        (len(trajectory_ref), 2, 2)
    )  # 1 rotation matrix per observation

    vel_mag = np.linalg.norm(trajectory_ref[:, 5:7], axis=1)
    cos_rot = trajectory_ref[:, 5] / vel_mag
    sin_rot = trajectory_ref[:, 6] / vel_mag

    rotation_matrices[:, 0, 0] = cos_rot
    rotation_matrices[:, 0, 1] = sin_rot
    rotation_matrices[:, 1, 0] = -sin_rot
    rotation_matrices[:, 1, 1] = cos_rot

    transformed_ref = trajectory_ref.copy()
    # translate the position to have it always at 0, 0
    transformed_ref[:, 1:3] -= trajectory_ref[:, 1:3]
    # translate the velocities
    transformed_ref[:, 5:7] -= trajectory_ref[:, 5:7]

    transformed_trajectories = []
    for trajectory in trajectories:
        transformed_trajectory = trajectory.copy()
        # transform the trajectory
        pos = transformed_trajectory[:, 1:3]
        pos -= trajectory_ref[:, 1:3]
        rotated_pos = np.diagonal(np.dot(rotation_matrices, pos.T), axis1=0, axis2=2).T
        transformed_trajectory[:, 1:3] = rotated_pos

        # transform the velocities
        vel = transformed_trajectory[:, 5:7]
        vel -= trajectory_ref[:, 5:7]
        rotated_vel = np.diagonal(np.dot(rotation_matrices, vel.T), axis1=0, axis2=2).T
        transformed_trajectory[:, 5:7] = rotated_vel
        transformed_trajectories += [transformed_trajectory]

    return transformed_ref, transformed_trajectories


def compute_observed_minimum_distance(
    trajectory: np.ndarray, interpolate: bool = False
) -> float:
    """Compute the observed minimum distance between the trajectory and the origin. Possibility
    to interpolate with the velocities to improve accuracy.

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    interpolate : bool, optional
        Whether or not to interpolate using the velocities, by default False

    Returns
    -------
    float
        The observed minimum distance
    """
    pos = trajectory[:, 1:3]
    vel = trajectory[:, 5:7]

    d = np.linalg.norm(pos, axis=1)
    if not interpolate:
        return np.min(d)
    else:
        # use velocities to interpolate the minimum distance:
        # project the position vector NO (from non group to group, i.e origin)
        # onto the line directed by
        # the velocity vector to get the distance to the point P for which the distance
        # between the line directed by the velocity and the origin
        # is smallest
        v_magn = np.linalg.norm(vel, axis=1)
        # print(v_magn)
        lambdas = (
            -np.matmul(vel, pos.T).diagonal() / v_magn
        )  # the diagonal of the matrix contains the dot products
        # compute the time to reach the point P
        t_to_P = (lambdas / v_magn)[:-1]  # last point is not used
        delta_ts = (trajectory[1:, 0] - trajectory[:-1, 0]) / 1000
        ids_interpolate_possible = np.where(
            np.logical_and(t_to_P < delta_ts, t_to_P >= 0)
        )[0]
        # ids_interpolate_possible = np.where(np.logical_and(t_to_P < 500, t_to_P >= 0))[
        #     0
        # ]
        d_interpolated = (
            d[ids_interpolate_possible] ** 2 - lambdas[ids_interpolate_possible] ** 2
        ) ** 0.5

        if len(d_interpolated):
            return min(np.min(d_interpolated), np.min(d))
        else:
            return np.min(d)


def compute_straight_line_minimum_distance(
    trajectory: np.ndarray, vicinity: int = 4000
) -> float:
    """Compute the distance between the origin and the straight line going through the entrance and exit
    of the vicinity (i.e. the positions with x closer to +vicinity and -vicinity).

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    vicinity : int, optional
        The size of the squared vicinity to be used, by default 4000

    Returns
    -------
    float
        The straight line distance
    """
    pos = trajectory[:, 1:3]
    trajectory_in_vicinity = trajectory[
        np.logical_and(np.abs(pos[:, 0]) <= vicinity, np.abs(pos[:, 1]) <= vicinity)
    ]
    pos_in_vicinity = trajectory_in_vicinity[:, 1:3]

    if len(trajectory_in_vicinity) <= 2:
        # do not get close enough
        return None

    idx_first = np.argmin(np.abs(pos_in_vicinity[:, 0] - vicinity))
    first_pos_in_vicinity = pos_in_vicinity[idx_first, :]
    idx_last = np.argmin(np.abs(pos_in_vicinity[:, 0] + vicinity))
    last_pos_in_vicinity = pos_in_vicinity[idx_last, :]

    # compute the distance between the line
    # from first F to last L and the point O at 0,0
    # i.e. (FL x OF) / ||OF||

    distance_to_straight_line = np.abs(
        cross(last_pos_in_vicinity - first_pos_in_vicinity, first_pos_in_vicinity)
    ) / np.linalg.norm(last_pos_in_vicinity - first_pos_in_vicinity)

    return distance_to_straight_line


def compute_straight_line_minimum_distance_from_vel(
    trajectory: np.ndarray, vicinity: int = 4000, n_points_average: int = 4
) -> float:
    """Compute the distance between the origin and the straight line going through the entrance and exit
    of the vicinity (i.e. the positions with x closer to +vicinity and -vicinity).

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    vicinity : int, optional
        The size of the squared vicinity to be used, by default 4000

    Returns
    -------
    float
        The straight line distance
    """
    pos = trajectory[:, 1:3]
    trajectory_in_vicinity = trajectory[
        np.logical_and(np.abs(pos[:, 0]) <= vicinity, np.abs(pos[:, 1]) <= vicinity)
    ]
    pos_in_vicinity = trajectory_in_vicinity[:, 1:3]

    if len(trajectory_in_vicinity) <= 2:
        # do not get close enough
        return None

    idx_first = np.argmin(np.abs(pos_in_vicinity[:, 0] - vicinity))
    first_pos_in_vicinity = pos_in_vicinity[idx_first, :]
    # first_vel_in_vicinity = trajectory_in_vicinity[idx_first, 5:7]

    vel_start_vicinity = np.nanmean(
        trajectory_in_vicinity[idx_first : idx_first + 1 + n_points_average, 5:7],
        axis=0,
    )  # average vel over some points

    # compute the distance from the line directed by the velocity to the origin
    # i.e. (v x OF) / ||OF||

    distance_to_straight_line = np.abs(
        cross(vel_start_vicinity, first_pos_in_vicinity)
    ) / np.linalg.norm(vel_start_vicinity)

    # distance_to_straight_line = np.abs(
    #     np.cross(vel_before, first_pos_in_vicinity)
    # ) / np.linalg.norm(vel_before)

    return distance_to_straight_line


def compute_alone_encounters(
    encounters: list[Pedestrian], pedestrian: Pedestrian, proximity_threshold: float
):
    """Filter a list of encounters to retain only the one where the encountered pedestrian
    is alone in the vicinity of the considered pedestrian.

    Parameters
    ----------
    encounters : list[Pedestrian]
        A list of encountered pedestrians
    pedestrian : Pedestrian
        A pedestrian for which the encountered are computed
    proximity_threshold : float
        The size of the vicinity to consider

    Returns
    -------
    _type_
        _description_
    """
    alone_encounters = []
    for pedestrian_A in encounters:
        alone = True
        for pedestrian_B in encounters:
            if pedestrian_A.get_id() == pedestrian_B.get_id():
                continue
            [traj_ped, traj_ped_A, traj_ped_B,] = compute_simultaneous_observations(
                [
                    pedestrian.get_trajectory(),
                    pedestrian_A.get_trajectory(),
                    pedestrian_B.get_trajectory(),
                ]
            )
            d_A = compute_interpersonal_distance(traj_ped, traj_ped_A)
            d_B = compute_interpersonal_distance(traj_ped, traj_ped_B)

            traj_with_A = traj_ped[d_A < proximity_threshold]
            traj_with_B = traj_ped[d_B < proximity_threshold]

            # plot_animated_2D_trajectories([traj_with_A, traj_with_B])

            if (
                len(compute_simultaneous_observations([traj_with_A, traj_with_B])[0])
                > len(traj_with_A) / 2
            ):
                alone = False
                break
        if alone:
            alone_encounters += [pedestrian_A]

    return alone_encounters


def compute_not_alone_encounters(
    encounters: list[Pedestrian], pedestrian: Pedestrian, proximity_threshold: float
):
    """Filter a list of encounters to retain only the one where the encountered pedestrian
    is not alone in the vicinity of the considered pedestrian.

    Parameters
    ----------
    encounters : list[Pedestrian]
        A list of encountered pedestrians
    pedestrian : Pedestrian
        A pedestrian for which the encountered are computed
    proximity_threshold : float
        The size of the vicinity to consider

    Returns
    -------
    _type_
        _description_
    """
    not_alone_encounters = []
    for pedestrian_A in encounters:
        alone = True
        for pedestrian_B in encounters:
            if pedestrian_A.get_id() == pedestrian_B.get_id():
                continue
            [traj_ped, traj_ped_A, traj_ped_B,] = compute_simultaneous_observations(
                [
                    pedestrian.get_trajectory(),
                    pedestrian_A.get_trajectory(),
                    pedestrian_B.get_trajectory(),
                ]
            )
            d_A = compute_interpersonal_distance(traj_ped, traj_ped_A)
            d_B = compute_interpersonal_distance(traj_ped, traj_ped_B)

            traj_with_A = traj_ped[d_A < proximity_threshold]
            traj_with_B = traj_ped[d_B < proximity_threshold]

            # plot_animated_2D_trajectories([traj_with_A, traj_with_B])

            if (
                len(compute_simultaneous_observations([traj_with_A, traj_with_B])[0])
                > len(traj_with_A) / 2
            ):
                alone = False
                break
        if not alone:
            not_alone_encounters += [pedestrian_A]

    return not_alone_encounters
