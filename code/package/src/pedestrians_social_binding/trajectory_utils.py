from __future__ import annotations
from typing import TYPE_CHECKING
from scipy.interpolate import CubicSpline
from scipy.integrate import trapz
from scipy.signal import savgol_filter, periodogram, find_peaks, hilbert
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from scipy.stats import entropy, circmean, circvar, circstd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import random

import pywt
from pycwt import wct, cwt, xwt


from pyrqa.analysis_type import Cross
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation


if TYPE_CHECKING:  # Only imports the below statements during type checking
    from pedestrians_social_binding.group import Group
    from pedestrians_social_binding.pedestrian import Pedestrian
    from pedestrians_social_binding.threshold import Threshold

from pedestrians_social_binding.constants import *
from pedestrians_social_binding.parameters import *


# from pedestrians_social_binding.plot_utils import plot_static_2D_trajectories

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
    trajectory_A: np.ndarray,
    trajectory_B: np.ndarray,
    rel_dir_angle_cos: float = REL_DIR_ANGLE_COS,
    rel_dir_min_perc: float = REL_DIR_MIN_PERC,
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
    # v_A = sim_traj_A[:, 5:7]
    # v_B = sim_traj_B[:, 5:7]

    v_A = pos_A[1:] - pos_A[:-1]
    v_B = pos_B[1:] - pos_B[:-1]

    # dot product of vA and vB
    v_d_dot = np.sum(v_A * v_B, axis=1)

    norm_product = np.linalg.norm(v_A, axis=1) * np.linalg.norm(v_B, axis=1)
    norm_product[norm_product == 0] = np.nan

    cos_vA_vB = v_d_dot / norm_product

    n_val = len(cos_vA_vB)

    n_same = np.sum(cos_vA_vB > rel_dir_angle_cos)
    n_opposite = np.sum(cos_vA_vB < -rel_dir_angle_cos)
    n_cross = n_val - n_same - n_opposite

    if n_same > rel_dir_min_perc * n_val:
        return "same"
    elif n_opposite > rel_dir_min_perc * n_val:
        return "opposite"
    elif n_cross > rel_dir_min_perc * n_val:
        return "cross"
    else:
        return None


def compute_trajectory_direction(trajectory: np.ndarray) -> str:
    """Compute the direction of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory

    Returns
    -------
    str
        The value of the relative direction ("left", "right")
    """
    # velocity = trajectory[:, 5:7]
    dp = np.diff(trajectory[:, 1:3], axis=0)
    dt = np.diff(trajectory[:, 0], axis=0)
    velocity = dp / dt[:, None]
    velocity_angle = np.arctan2(velocity[:, 1], velocity[:, 0])
    velocity_angle[velocity_angle > np.pi] -= 2 * np.pi
    velocity_angle[velocity_angle < -np.pi] += 2 * np.pi

    mask_left = np.abs(velocity_angle) > 7 * np.pi / 8
    mask_right = np.abs(velocity_angle) < np.pi / 8
    n_left = np.sum(mask_left)
    n_right = np.sum(mask_right)
    perc_left = n_left / len(velocity_angle)
    perc_right = n_right / len(velocity_angle)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[1].plot(np.abs(velocity_angle))
    # ax[1].set_ylim(0, np.pi)

    # plot_static_2D_trajectories(
    #     [trajectory],
    #     gradient=True,
    #     boundaries=env.boundaries,
    #     ax=ax[0],
    #     show=False,
    # )
    # plt.show()

    # print(
    #     f"Pedestrian {pedestrian_id} is {perc_left:.2f} left and {perc_right:.2f} right"
    # )

    if perc_left > DIR_MIN_PERC:
        return "left"
    elif perc_right > DIR_MIN_PERC:
        return "right"
    else:
        return None


def compute_encounter_side(trajectory_A: np.ndarray, trajectory_B: np.ndarray) -> str:
    """Compute the encounter side between two trajectories. First, compute
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
        The value of the encounter side ("left", "right")
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

    # check if B is mostly on the right of A
    # d_AB is the vector from A to B
    B_on_right = np.cross(v_A, pos_B - pos_A) < 0
    n_val = len(B_on_right)
    n_right = np.sum(B_on_right)
    n_left = n_val - n_right

    if n_right > ENC_SIDE_MIN_PERC * n_val:
        return "right"
    elif n_left > ENC_SIDE_MIN_PERC * n_val:
        return "left"
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

    # elif value == "t":  # threshold on the time
    #     time = pedestrian.get_time()
    #     t_obs = time[-1] - time[0]
    #     if (
    #         (
    #             min_val is not None
    #             and max_val is not None
    #             and t_obs <= max_val
    #             and t_obs >= min_val
    #         )
    #         or (min_val is not None and t_obs >= min_val)
    #         or (max_val is not None and t_obs <= max_val)
    #     ):
    #         return pedestrian
    #     else:
    #         return None

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
    elif value in ["x", "y", "t"]:  # threshold on the position
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


def compute_continuous_sub_trajectories_using_time(
    trajectory: np.ndarray, max_gap: float = 2
) -> list[np.ndarray]:
    """Breaks down a trajectory in to a list of sub-trajectories that have maximum time
    gaps of max_gap

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    max_gap : int, optional
        The maximum temporal gap allowed in a trajectory, by default 2

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


def compute_continuous_sub_trajectories_using_distance(
    trajectory: np.ndarray, max_distance: int = 5000, min_length: int = 5
) -> tuple[list[np.ndarray], list[float]]:
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
    liste_of_length = []

    for s in range(len(trajectory)):
        for i in range(s + 1, len(trajectory)):
            delta = trajectory[i, 1:3] - trajectory[s, 1:3]
            distance = np.sqrt(np.sum(delta**2))
            if distance >= max_distance:
                if i - s >= min_length:
                    sub_sub_trajectories += [trajectory[s:i, :]]
                    liste_of_length += [distance]
                    break

    if len(sub_sub_trajectories) == 0:
        return None

    return sub_sub_trajectories, liste_of_length


def compute_continuous_sub_trajectories_using_distance_v2(
    trajectory: np.ndarray, max_distance: int = 5000, min_length: int = 5
) -> tuple[list[np.ndarray], list[float]]:
    s = 0

    sub_sub_trajectories = []
    liste_of_length = []

    for s in range(0, len(trajectory), min_length):
        if s + min_length >= len(trajectory):
            break
        delta = trajectory[s + min_length, 1:3] - trajectory[s, 1:3]
        distance = np.sqrt(np.sum(delta**2))

        sub_sub_trajectories += [trajectory[s : s + min_length, :]]
        liste_of_length += [distance]

    if len(sub_sub_trajectories) == 0:
        return None

    return sub_sub_trajectories, liste_of_length


def compute_turning_angle_integral_spline(trajectory: np.ndarray):
    cs = CubicSpline(trajectory[:, 0], trajectory[:, 1:3])
    times = np.linspace(trajectory[0, 0], trajectory[-1, 0], SAMPLING_NUMBER)
    interpolation = cs(times)
    turning_angles = compute_turning_angles(interpolation)
    # plt.plot(times[1:-1], turning_angles)
    # plt.show()
    angle_integral = trapz(np.abs(turning_angles), times[1:-1])
    return angle_integral


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
    n_average=66,
    plot: bool = False,
    ax: plt.Axes | None = None,
) -> float:
    """Computes the maximum lateral deviation over the trajectory (the maximum distance from points of the trajectories to the line joining the first and last point of the trajectory).

    Parameters
    ----------
    traj : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to average the velocity over, by default 3
    Returns
    -------
    float
        The value for the maximum lateral deviation
    """
    pos = traj[:, 1:3]
    dt = (traj[1:, 0] - traj[:-1, 0]) / 1000
    dp = pos[1:] - pos[:-1]
    vel = dp / dt[:, None]

    start_point = pos[0]
    middle_points = pos[1:-1]

    start_vel = np.nanmean(vel[:n_average], axis=0)

    distances_to_straight_line = np.abs(
        cross(start_vel, middle_points - start_point)
    ) / np.linalg.norm(start_vel)

    idx_max = np.argmax(distances_to_straight_line) + 1
    position_max = pos[np.argmax(distances_to_straight_line) + 1, :]

    max_distance = np.max(distances_to_straight_line) / 1000

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        # show trajectory
        ax.plot(pos[:, 0] / 1000, pos[:, 1] / 1000, color="black")
        # show line guided by velocity
        end_line = start_point + start_vel * (
            np.dot(start_vel, (pos[-1] - start_point)) / np.dot(start_vel, start_vel)
        )
        ax.plot(
            [start_point[0] / 1000, end_line[0] / 1000],
            [start_point[1] / 1000, end_line[1] / 1000],
            color="red",
        )
        # show furthest point in red
        ax.scatter(
            position_max[0] / 1000,
            position_max[1] / 1000,
            color="red",
            s=20,
        )
        # show distance to line in purple
        point_on_line_furthest = start_point + start_vel * (
            np.dot(start_vel, (position_max - start_point))
            / np.dot(start_vel, start_vel)
        )
        ax.plot(
            [position_max[0] / 1000, point_on_line_furthest[0] / 1000],
            [position_max[1] / 1000, point_on_line_furthest[1] / 1000],
            color="purple",
        )
        ax.set_aspect("equal")

    return max_distance  # , idx_max


def compute_maximum_lateral_deviation_using_vel_with_interpolation(
    traj: np.ndarray,
    n_average=3,
    interpolate: bool = False,
) -> float:
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


def compute_signed_maximum_lateral_deviation_using_vel(
    traj: np.ndarray,
    right_encounter: bool = True,
    n_average: int = 66,
    plot: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[float, int]:
    """Computes the maximum lateral deviation over the trajectory (the maximum distance from points of the trajectories to the line joining the first and last point of the trajectory).

    Parameters
    ----------
    traj : np.ndarray
        A trajectory
    right_encounter : bool, optional
        Whether the encounter is a right encounter or not, by default True
    n_average : int, optional
        The number of points to average the velocity over, by default 3
    plot : bool, optional
        Whether or not to plot the result, by default False
    ax : plt.Axes | None, optional
        The axes on which to plot, by default None
    Returns
    -------
    float
        The value for the maximum lateral deviation
    """
    pos = traj[:, 1:3]
    delta_ts = (traj[1:, 0] - traj[:-1, 0]) / 1000
    delta_ps = pos[1:] - pos[:-1]
    vel = delta_ps / delta_ts[:, None]

    start_point = pos[0]
    middle_points = pos[1:-1]

    start_vel = np.nanmean(vel[:n_average], axis=0)

    distances_to_straight_line = np.abs(
        cross(start_vel, middle_points - start_point)
    ) / np.linalg.norm(start_vel)

    idx_max = np.argmax(distances_to_straight_line) + 1
    position_max = pos[np.argmax(distances_to_straight_line) + 1, :]

    max_distance = np.max(distances_to_straight_line)

    # check if the furthest point is on the right or left of the line
    start_line = start_point
    end_line = start_point + start_vel
    point_max = pos[idx_max]
    max_on_right = np.cross(end_line - start_line, point_max - start_line) < 0
    # sign the distance
    if right_encounter:
        if max_on_right:
            max_distance *= -1
    else:
        if not max_on_right:
            max_distance *= -1

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        # show line guided by velocity
        end_line = start_point + start_vel * (
            np.dot(start_vel, (pos[-1] - start_point)) / np.dot(start_vel, start_vel)
        )
        ax.plot(
            [start_point[0] / 1000, end_line[0] / 1000],
            [start_point[1] / 1000, end_line[1] / 1000],
            color="red",
        )
        # show furthest point in red
        ax.scatter(
            position_max[0] / 1000,
            position_max[1] / 1000,
            color="red",
            s=20,
        )
        # show distance to line in purple
        point_on_line_furthest = start_point + start_vel * (
            np.dot(start_vel, (position_max - start_point))
            / np.dot(start_vel, start_vel)
        )
        ax.plot(
            [position_max[0] / 1000, point_on_line_furthest[0] / 1000],
            [position_max[1] / 1000, point_on_line_furthest[1] / 1000],
            color="purple",
        )

    return max_distance, idx_max


def compute_maximum_lateral_deviation_using_all_vel(
    traj: np.ndarray,
    plot: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[float, int]:
    """Computes the maximum lateral deviation over the trajectory (the maximum distance from points of the trajectories to the line joining the first and last point of the trajectory).

    Parameters
    ----------
    traj : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to average the velocity over, by default 3
    Returns
    -------
    float
        The value for the maximum lateral deviation
    """
    pos = traj[:, 1:3]
    delta_ts = (traj[1:, 0] - traj[:-1, 0]) / 1000
    delta_ps = pos[1:] - pos[:-1]
    vel = delta_ps / delta_ts[:, None]

    start_point = pos[0]
    middle_points = pos[1:-1]

    start_vel = np.nanmean(vel, axis=0)

    distances_to_straight_line = np.abs(
        cross(start_vel, middle_points - start_point)
    ) / np.linalg.norm(start_vel)

    idx_max = np.argmax(distances_to_straight_line) + 1
    position_max = pos[np.argmax(distances_to_straight_line) + 1, :]

    max_distance = np.max(distances_to_straight_line)

    if plot and ax is not None:
        # show line guided by velocity
        end_line = start_point + start_vel * (
            np.dot(start_vel, (pos[-1] - start_point)) / np.dot(start_vel, start_vel)
        )
        ax.plot(
            [start_point[0] / 1000, end_line[0] / 1000],
            [start_point[1] / 1000, end_line[1] / 1000],
            color="red",
        )
        # show furthest point in red
        ax.scatter(
            position_max[0] / 1000,
            position_max[1] / 1000,
            color="red",
            s=20,
        )
        # show distance to line in purple
        point_on_line_furthest = start_point + start_vel * (
            np.dot(start_vel, (position_max - start_point))
            / np.dot(start_vel, start_vel)
        )
        ax.plot(
            [position_max[0] / 1000, point_on_line_furthest[0] / 1000],
            [position_max[1] / 1000, point_on_line_furthest[1] / 1000],
            color="purple",
        )
        # print("fu")

    return max_distance, idx_max


def compute_maximum_lateral_deviation_spline(traj: np.ndarray, n_average: int = 2):
    pos = traj[:, 1:3]
    start_point = pos[0]

    start_vel = np.nanmean(traj[:n_average, 5:7], axis=0)

    # print(traj, start_vel)
    cs = CubicSpline(traj[:, 0], traj[:, 1:3])
    xs = np.linspace(traj[0, 0], traj[-1, 0], SAMPLING_NUMBER)
    interpolation = cs(xs)
    middle_points = interpolation[1:]

    distances_to_straight_line = np.abs(
        cross(start_vel, middle_points - start_point)
    ) / np.linalg.norm(start_vel)

    max_distance = np.max(distances_to_straight_line)
    idx_max = np.argmax(distances_to_straight_line) + 1
    return max_distance, idx_max


def compute_maximum_lateral_deviation_using_vel_2(
    traj: np.ndarray,
    n_average=3,
    interpolate: bool = False,
    length: float = None,
) -> dict[
    "max_lateral_deviation":float,
    "position of max lateral deviation" : np.ndarray,
    "mean_velocity" : np.ndarray,
    "length_of_trajectory":float,
]:
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
    dict_return = {
        "max_lateral_deviation": 0,
        "position of max lateral deviation": np.array([0, 0, 0, 0, 0, 0, 0]),
        "mean_velocity": np.array([0, 0]),
        "length_of_trajectory": 0,
    }

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
        dict_return["position of max lateral deviation"] = traj[
            np.argmax(distances_to_straight_line) + 1, :
        ]
        dict_return["mean_velocity"] = start_vel
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
        dict_return["position of max lateral deviation"] = traj[
            np.argmax(distances_to_straight_line), :
        ]
        dict_return["mean_velocity"] = start_vel
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
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

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
    # not more that 2 columns
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

    net_displacement = compute_net_displacement(position)
    gross_displacement = compute_gross_displacement(position)
    return net_displacement / gross_displacement


def compute_max_straightness_index(position: np.ndarray) -> float:
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

    max_straightness_index = 0
    for i in range(1, len(position)):
        net_displacement = compute_net_displacement(position[:i])
        gross_displacement = compute_gross_displacement(position[:i])
        straightness_index = 1 - net_displacement / gross_displacement
        if straightness_index > max_straightness_index:
            max_straightness_index = straightness_index
    return max_straightness_index


def compute_straightness_index_spline(traj: np.ndarray) -> float:
    cs = CubicSpline(traj[:, 0], traj[:, 1:3])
    xs = np.linspace(traj[0, 0], traj[-1, 0], SAMPLING_NUMBER)
    interpolation = cs(xs)

    # show spline and trajectory
    # plt.scatter(
    #     traj[:, 1],
    #     traj[:, 2],
    #     label="data",
    # )
    # plt.plot(interpolation[:, 0], interpolation[:, 1], label="spline")
    # plt.axis("equal")
    # plt.show()

    net_displacement = compute_net_displacement(interpolation)
    gross_displacement = compute_gross_displacement(interpolation)
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
    # not more that 2 columns
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

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


def rediscretize_trajectory(
    trajectory: np.ndarray, step_length: None | float = None
) -> np.ndarray:
    """Transforms the trajectory so that the distance between each point is fixed

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    step_length : None | float, optional
        The length of the step, by default None

    Returns
    -------
    np.ndarray
        The trajectory with a constant step size
    """

    rediscretized_position = []
    current_point = trajectory[0, 0:3]
    start_segment, end_segment = trajectory[0, 0:3], trajectory[1, 0:3]
    start_segment_index, end_segment_index = 0, 1
    rediscretized_position += [current_point]
    done = False
    while not done:
        d_point_goal = np.linalg.norm(end_segment[1:3] - current_point[1:3])
        if d_point_goal > step_length:  # there is space until the next trajectory point
            l = step_length / d_point_goal
            current_point = l * end_segment + (1 - l) * current_point
            rediscretized_position += [current_point]
        else:  # no more space, the points needs to be on one of the next segments
            # find the closest segment that intersects the circle centered on current_point
            # with radius step_length
            found_intersection = False
            while not found_intersection:
                if end_segment_index == len(trajectory) - 1:  # last point
                    done = True
                    break
                start_segment_index = end_segment_index
                end_segment_index = start_segment_index + 1
                start_segment = trajectory[start_segment_index, :3]
                end_segment = trajectory[end_segment_index, :3]
                intersection = find_intersection_circle_segment_with_time(
                    current_point,
                    step_length,
                    start_segment,
                    end_segment,
                )
                if intersection is not None:
                    found_intersection = True
                    current_point = intersection
                    rediscretized_position += [current_point]

    rediscretized_position = np.array(rediscretized_position)
    rediscretized_trajectory = np.zeros((len(rediscretized_position), 7))
    rediscretized_trajectory[:, 0:3] = rediscretized_position
    return rediscretized_trajectory


def rediscretize_position_v2(position: np.ndarray, step_length: float):
    """Transforms the trajectory so that the distance between each point is fixed

    Parameters
    ----------
    position : np.ndarray
        A position
    step_length : float
        The length of the step
    """

    rediscretized_position = []
    current_point = position[0]
    start_segment, end_segment = position[0], position[1]
    start_segment_index, end_segment_index = 0, 1
    rediscretized_position += [current_point]
    done = False
    while not done:
        d_point_goal = np.linalg.norm(end_segment - current_point)
        if d_point_goal > step_length:  # there is space until the next trajectory point
            l = step_length / d_point_goal
            current_point = l * end_segment + (1 - l) * current_point
            rediscretized_position += [current_point]
        else:  # no more space, the points needs to be on one of the next segments
            # find the closest segment that intersects the circle centered on current_point
            # with radius step_length
            found_intersection = False
            while not found_intersection:
                if end_segment_index == len(position) - 1:  # last point
                    done = True
                    break
                start_segment_index = end_segment_index
                end_segment_index = start_segment_index + 1
                start_segment = position[start_segment_index]
                end_segment = position[end_segment_index]
                intersection = find_intersection_circle_segment(
                    current_point,
                    step_length,
                    start_segment,
                    end_segment,
                )
                if intersection is not None:
                    found_intersection = True
                    current_point = intersection
                    rediscretized_position += [current_point]

    rediscretized_position = np.array(rediscretized_position)
    # diff = np.linalg.norm(
    #     rediscretized_position[:-1] - rediscretized_position[1:], axis=1
    # )
    # print(diff)
    # plt.scatter(position[:, 0], position[:, 1])
    # plt.plot(
    #     np.array(rediscretized_position)[:, 0],
    #     np.array(rediscretized_position)[:, 1],
    #     "ro-",
    # )
    # plt.axis("equal")
    # plt.show()

    return rediscretized_position


def find_intersection_circle_segment(C, radius, A, B):
    # plot
    # fig, ax = plt.subplots()
    # ax.scatter(C[0], C[1])
    # ax.scatter(A[0], A[1])
    # ax.scatter(B[0], B[1])
    # ax.plot([A[0], B[0]], [A[1], B[1]])
    # circle = patches.Circle((C[0], C[1]), radius, color="r", fill=False)
    # ax.add_patch(circle)
    # ax.axis("equal")
    # plt.show()

    AC = C - A
    AB = B - A
    AB_norm = np.linalg.norm(AB)
    AC_norm = np.linalg.norm(AC)
    # (x, y) is on the circle if (x - Cx)^2 + (y - Cy)^2 = radius^2
    # (x, y) is on the segment if AP = lambda * AB with 0 <= lambda <= 1
    # solve quadratic equation for lambda
    a = AB_norm**2
    b = 2 * np.dot(-AC, AB)
    c = AC_norm**2 - radius**2
    delta = b**2 - 4 * a * c
    if a == 0:
        if b == 0:
            return None
        lambda_1 = -c / b
        if 0 <= lambda_1 <= 1:
            return A + lambda_1 * AB
        else:
            return None
    if delta < 0:
        return None
    lambda_1 = (-b - delta**0.5) / (2 * a)
    lambda_2 = (-b + delta**0.5) / (2 * a)
    # is one of the lambdas between 0 and 1?
    if 0 <= lambda_1 <= 1:
        return A + lambda_1 * AB
    elif 0 <= lambda_2 <= 1:
        return A + lambda_2 * AB
    else:
        return None


def find_intersection_circle_segment_with_time(C, radius, A, B):
    AC = C - A
    AB = B - A
    AB_norm = np.linalg.norm(AB[1:3])
    AC_norm = np.linalg.norm(AC[1:3])
    # (x, y) is on the circle if (x - Cx)^2 + (y - Cy)^2 = radius^2
    # (x, y) is on the segment if AP = lambda * AB with 0 <= lambda <= 1
    # solve quadratic equation for lambda
    a = AB_norm**2
    b = 2 * np.dot(-AC[1:3], AB[1:3])
    c = AC_norm**2 - radius**2
    delta = b**2 - 4 * a * c
    if a == 0:
        if b == 0:
            return None
        lambda_1 = -c / b
        if 0 <= lambda_1 <= 1:
            return A + lambda_1 * AB
        else:
            return None
    if delta < 0:
        return None
    lambda_1 = (-b - delta**0.5) / (2 * a)
    lambda_2 = (-b + delta**0.5) / (2 * a)
    # is one of the lambdas between 0 and 1?
    if 0 <= lambda_1 <= 1:
        return A + lambda_1 * AB
    elif 0 <= lambda_2 <= 1:
        return A + lambda_2 * AB
    else:
        return None


def rediscretize_position(
    position: np.ndarray, step_length: None | float = None
) -> np.ndarray:
    """Transforms the trajectory so that the distance between each point is fixed

    Parameters
    ----------
    position : np.ndarray
        A position
    step_length : None | float, optional
        The length of the step, by default None

    Returns
    -------
    np.ndarray
        The position with a constant step size
    """
    step_sizes = np.linalg.norm(position[:-1] - position[1:], axis=1)
    n_points = len(position)
    if step_length is None:
        q = np.min(step_sizes)
    else:
        q = step_length
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


def compute_curvature(trajectory: np.ndarray) -> list[int]:
    dt = trajectory[1:, 0] - trajectory[:-1, 0]
    dp = (trajectory[1:, 1:3] - trajectory[:-1, 1:3]) / 1000
    v = dp / dt[:, None]
    a = (v[1:, :] - v[:-1, :]) / dt[1:, None]
    v_mag = np.linalg.norm(v[1:, :], axis=1)
    non_zero = v_mag != 0
    k = np.cross(v[1:, :], a, axis=1)[non_zero] / v_mag[non_zero] ** 3
    return np.abs(k)


def compute_curvature_gradient(trajectory: np.ndarray) -> list[int]:
    v = np.gradient(trajectory[:, 1:3], trajectory[:, 0], axis=0)
    a = np.gradient(v, trajectory[:, 0], axis=0)
    v_mag = np.linalg.norm(v, axis=1)
    k = np.cross(v, a, axis=1) / v_mag**3
    return k


def compute_tangents(trajectory: np.ndarray) -> list[int]:
    v = np.gradient(trajectory[:, 1:3], trajectory[:, 0], axis=0)
    v_mag = np.linalg.norm(v, axis=1)
    t = v / v_mag[:, None]
    return t


def compute_normals(trajectory: np.ndarray) -> list[int]:
    t = compute_tangents(trajectory)
    n = np.zeros_like(t)
    n[:, 0] = -t[:, 1]
    n[:, 1] = t[:, 0]
    return n


def compute_curvature_unsigned(trajectory: np.ndarray) -> list[int]:
    dt = trajectory[1:, 0] - trajectory[:-1, 0]
    dp = (trajectory[1:, 1:3] - trajectory[:-1, 1:3]) / 1000
    v = dp / dt[:, None]
    a = (v[1:, :] - v[:-1, :]) / dt[1:, None]
    v_mag = np.linalg.norm(v[1:, :], axis=1)
    non_zero = v_mag != 0
    k = np.cross(v[1:, :], a, axis=1)[non_zero] / v_mag[non_zero] ** 3
    return k


def compute_curvature_integral(trajectory: np.ndarray) -> float:
    unsigned_curvature = compute_curvature_unsigned(trajectory)
    return np.abs(np.trapz(unsigned_curvature, trajectory[2:, 0]))


def compute_tortuosity_from_curvature(trajectory: np.ndarray) -> float:
    unsigned_curvature = compute_curvature_unsigned(trajectory)
    squared_curvature = np.array(unsigned_curvature) ** 2
    return np.trapz(squared_curvature, trajectory[2:, 0]) / compute_gross_displacement(
        trajectory[:, 1:3]
    )


def compute_curvature_spline(
    trajectory: np.ndarray, f_sampling: float = 100
) -> list[int]:
    cs = CubicSpline(trajectory[:, 0], trajectory[:, 1:3] / 1000)

    dcs = cs.derivative()
    ddcs = dcs.derivative()

    duration = trajectory[-1, 0] - trajectory[0, 0]
    n_points = int(duration * f_sampling)

    t = np.linspace(trajectory[0, 0], trajectory[-1, 0], n_points)

    v = dcs(t)
    a = ddcs(t)

    k = np.cross(v, a, axis=1) / np.linalg.norm(v, axis=1) ** 3

    # v = dcs(trajectory[:, 0])
    # a = ddcs(trajectory[:, 0])

    # if np.max(k) > 1:
    #     # show spline and trajectory
    # plt.scatter(trajectory[:, 1] / 1000, trajectory[:, 2] / 1000, label="data", s=4)
    # interpolation = cs(t)
    # plt.scatter(interpolation[:, 0], interpolation[:, 1], label="spline", c=k, s=1)
    # plt.axis("equal")
    # plt.show()

    return np.abs(k), v, a, t


def compute_curvature_v2(trajectory: np.ndarray) -> list[int]:
    pos = trajectory[:, 1:3]
    vel = trajectory[:, 5:7]
    delta_ts = (trajectory[1:, 0] - trajectory[:-1, 0]) / 1000

    extr_time = trajectory[:-1, 0] + delta_ts / 2
    extr_pos = pos[:-1] + delta_ts[:, None] / 2 * vel[:-1]
    extr_vel = vel[:-1] + delta_ts[:, None] / 2 * (vel[1:] - vel[:-1])
    extr_speed = np.linalg.norm(extr_vel, axis=1)

    # print("extr_time",np.shape(extr_time))
    # print("extr_pos",np.shape(extr_pos))
    extr_point = np.concatenate((extr_time[:, None], extr_pos), axis=1)
    # print(type(extr_point))
    # print("extra_point",np.shape(extr_point))
    extr_point = np.concatenate((extr_point, np.zeros((len(extr_point), 1))), axis=1)
    # print("extra_point1",np.shape(extr_point))
    extr_point = np.concatenate((extr_point, extr_speed[:, None]), axis=1)
    # print("extra_point2",np.shape(extr_point))
    extr_point = np.concatenate((extr_point, extr_vel), axis=1)
    # print("extra_point3",np.shape(extr_point))

    new_trajectory = np.ndarray(shape=(len(trajectory) + len(extr_point), 7))

    count = 0
    for i, traj in enumerate(trajectory):
        new_trajectory[count] = traj
        count += 1
        if i < len(trajectory) - 1:
            new_trajectory[count] = extr_point[i]
            count += 1

    # print("trajectory", (trajectory[1:, 0] + trajectory[:-1, 1:3])/2)
    # print("new_traj", new_trajectory[:, 0])

    v = new_trajectory[:, 5:7]
    a = v[1:, :] - v[:-1, :]
    a = a / 100
    k = np.cross(v[1:, :], a, axis=1) / np.linalg.norm(v[1:, :], axis=1) ** 3
    return np.abs(k)


def compute_sinuosity(
    position: np.ndarray, step_length: None | float = None, rediscretize=True
) -> float:
    """Computes the sinuosity of the trajectory. Sinuosity is defined as 1.18 * s/q where
    s is the standard deviation of the turning angles of the trajectory and q is the step size
    of the trajectory

    Parameters
    ----------
    position : np.ndarray
        A position
    step_length : None | float, optional
        The length of the step, by default None
    rediscretize : bool, optional
        Whether or not to rediscretize the trajectory, by default True

    Returns
    -------
    float
        The value for the sinuosity
    """

    # not more that 2 columns
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

    if rediscretize:
        position = rediscretize_position_v2(position, step_length=step_length)
        if len(position) <= 1:
            return np.nan
    step_length = np.linalg.norm(position[1] - position[0]) / 1000
    turning_angles = compute_turning_angles(position)
    # sinuosity = 1.18 * np.std(turning_angles) / step_length**0.5
    sinuosity = 1.18 * circstd(turning_angles, high=np.pi, low=-np.pi) / step_length
    return sinuosity


def compute_turning_angle_integral(
    position: np.ndarray, step_length: None | float = None
) -> float:
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

    rediscretized_position = rediscretize_position_v2(position, step_length=step_length)
    if len(rediscretized_position) <= 1:
        return np.nan
    turning_angles = compute_turning_angles(rediscretized_position)
    return np.abs(np.trapz(turning_angles))


def compute_max_cumulative_turning_angle(
    position: np.ndarray, step_length: None | float = None, rediscretize=True
) -> float:
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

    if rediscretize:
        position = rediscretize_position_v2(position, step_length=step_length)
        if len(position) <= 1:
            return np.nan
    turning_angles = compute_turning_angles(position)
    return np.max(np.abs(np.cumsum(turning_angles)))


def compute_vertical_maximum_deviation(trajectory: np.ndarray) -> float:
    """Computes the maximum vertical deviation over the trajectory (the maximum distance from points of the trajectories to the x axis)

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory

    Returns
    -------
    float
        The value for the maximum vertical deviation
    """
    initial_y = trajectory[0, 2]
    index_max_deviation = np.argmax(np.abs(trajectory[:, 2] - initial_y))
    maximum_vertical_deviation = np.abs(trajectory[index_max_deviation, 2] - initial_y)
    return maximum_vertical_deviation


def compute_sinuosity_without_rediscretization(position: np.ndarray) -> float:
    """Computes the sinuosity of the trajectory using the formula from Benhamou, S. (2004).

    Parameters
    ----------
    position : np.ndarray
        The positions

    Returns
    -------
    float
        The value for the sinuosity
    """
    # not more that 2 columns
    if position.shape[1] > 2:
        raise ValueError("Position should be a 2D array")

    step_lengths = np.linalg.norm(position[:-1, :] - position[1:, :], axis=1)
    p = np.mean(step_lengths)  # mean step length
    std_step_length = np.std(step_lengths)
    b = std_step_length / p  # coefficient of variation of step length
    turning_angles = compute_turning_angles(position)
    cos_turning_angles = np.cos(turning_angles)
    c = np.mean(cos_turning_angles)  # mean cosine of turning angles
    if c == 1:
        return 0
    sinuosity = 2 * (p * ((1 + c) / (1 - c) + b**2)) ** -0.5
    return sinuosity


def compute_sinuosity_spline(trajectory: np.ndarray) -> float:
    NUMBER_OF_POINTS = SAMPLING_NUMBER

    cs = CubicSpline(trajectory[:, 0], trajectory[:, 1:3])
    times = np.linspace(trajectory[0, 0], trajectory[-1, 0], NUMBER_OF_POINTS)
    interpolation = cs(times)
    rediscretized_position = rediscretize_position(interpolation)
    step_length = np.linalg.norm(rediscretized_position[1] - rediscretized_position[0])
    turning_angles = compute_turning_angles(rediscretized_position)
    sinuosity = 1.18 * np.std(turning_angles) / step_length**0.5
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
    # i.e. (SE x PE) / ||SE||
    distances_to_straight_line = np.abs(
        cross(end_point - start_point, position - start_point)
    ) / np.linalg.norm(end_point - start_point)
    # compte the integral using the trapezoid
    # compute the projection of the trajectory points onto the straight line
    # get the bases of the trapezoid
    distances_to_first_point = distance.cdist([start_point], position)[0]
    cumul_bases = (distances_to_first_point**2 - distances_to_straight_line**2) ** 0.5
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
    trajectory: np.ndarray,
    piece_size: int,
    overlap: bool = False,
    delta: int = 100,
    max_pieces: int | None = None,
) -> list[np.ndarray]:
    """Breaks up a trajectory in to pieces of given length

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
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
    position = trajectory[:, 1:3]
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
        pieces += [trajectory[start : end + 1, :]]

    if max_pieces is not None and len(pieces) > max_pieces:
        # sample random pieces
        pieces = random.sample(pieces, max_pieces)

    return pieces


def get_pieces_indices(
    trajectory: np.ndarray,
    piece_size: int | None,
    overlap: bool = False,
    delta: int = 100,
    max_pieces: int | None = None,
) -> list[np.ndarray]:
    """Breaks up a trajectory in to pieces of given length

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
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
    position = trajectory[:, 1:3]
    start = position[0]
    end = position[1]

    points_distances = np.triu(distance.cdist(position, position, "euclidean"), k=0)

    indices_smaller = np.argwhere(np.abs(points_distances - piece_size) < delta)

    indices = []
    current_min = 0
    for start, end in indices_smaller:
        # check if there is some overlaping, and skip piece
        if not overlap and start < current_min:
            continue
        current_min = end

        # get the trajectory between these points
        indices += [[start, end]]

    if max_pieces is not None and len(indices) > max_pieces:
        # sample random pieces
        indices = random.sample(indices, max_pieces)

    return indices


def get_pieces_indices_from_time(
    trajectory: np.ndarray,
    piece_duration: int,
    overlap: bool = False,
    delta: int = 0.2,
    max_pieces: int | None = None,
) -> list[np.ndarray]:
    """Breaks up a trajectory in to pieces of given length

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    piece_duration : int
        The duration of the pieces
    overlap : bool, optional
        Whether or not overlapping pieces should be returned, by default False
    delta : int, optional
        The maximum difference allowed between the duration of the pieces
        requested and the pieces returned, by default 2

    Returns
    -------
    list[np.ndarray]
        The list of pieces
    """
    times = trajectory[:, 0]
    start = times[0]
    end = times[1]

    t_times = np.reshape(times, (len(times), 1))
    time_distances = t_times.T - t_times

    indices_smaller = np.argwhere(np.abs(time_distances - piece_duration) < delta)

    indices = []
    current_min = 0

    for start, end in indices_smaller:
        # check if there is some overlaping, and skip piece
        if not overlap and start < current_min:
            continue
        current_min = end

        # get the trajectory between these points
        indices += [[start, end]]

    if max_pieces is not None and len(indices) > max_pieces:
        # sample random pieces
        indices = random.sample(indices, max_pieces)

    return indices


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


def resample_trajectory(
    trajectory: np.ndarray, sampling_time: float = 0.5, interpolation: str = "linear"
) -> np.ndarray:
    """Resample the trajectory using the given time step

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    sampling_time : float, optional
        The new sampling time, by default 500
    interpolation : str, optional
        The interpolation method to use, by default "linear"
    Returns
    -------
    np.ndarray
        The trajectory sampled a intervals of given time
    """
    times = trajectory[:, 0]
    duration = times[-1] - times[0]
    n_points = int(duration / sampling_time)
    new_times = np.linspace(times[0], times[-1], n_points)
    new_trajectory = np.ndarray(shape=(n_points, 7))
    new_trajectory[:, 0] = new_times
    if interpolation == "linear":
        # linear interpolation for the position
        new_trajectory[:, 1] = np.interp(new_times, times, trajectory[:, 1])
        new_trajectory[:, 2] = np.interp(new_times, times, trajectory[:, 2])
    elif interpolation == "spline":
        # spline interpolation for the position
        # remove duplicate times
        unique_times, unique_indices = np.unique(times, return_index=True)
        unique_positions = trajectory[unique_indices, 1:3]
        cs = CubicSpline(unique_times, unique_positions)
        new_trajectory[:, 1:3] = cs(new_times)

    return new_trajectory


# def resample_trajectory(trajectory: np.ndarray, sampling_time: int = 500) -> np.ndarray:
#     """Resample the trajectory using the given time step

#     Parameters
#     ----------
#     trajectory : np.ndarray
#         A trajectory
#     sampling_time : int, optional
#         The time step to use (in ms), by default 500

#     Returns
#     -------
#     np.ndarray
#         The trajectory sampled a intervals of given time
#     """
#     current_time_step = get_time_step(trajectory)
#     if (
#         current_time_step < 1.2 * sampling_time
#         and current_time_step > 0.8 * sampling_time
#     ):  # nothing to do
#         return trajectory

#     times = trajectory[:, 0]
#     traj_time_steps = times[1:] - times[:-1]

#     rows_to_keep = [0]
#     current_step = 0
#     for i, time_step in enumerate(traj_time_steps):
#         if current_step >= sampling_time:
#             rows_to_keep += [i]
#             current_step = 0
#         current_step += time_step
#     return trajectory[rows_to_keep, ...]


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
    trajectory_ref: np.ndarray,
    trajectories: list[np.ndarray],
    axis: str = "x",
    nullify_velocities: bool = True,
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
    simultaneous_trajectories = compute_simultaneous_observations(
        [trajectory_ref] + trajectories
    )
    trajectory_ref = simultaneous_trajectories[0]
    trajectories = simultaneous_trajectories[1:]

    # build the rotation matrix
    rotation_matrices = np.zeros(
        (len(trajectory_ref), 2, 2)
    )  # 1 rotation matrix per observation

    vel_mag = np.linalg.norm(trajectory_ref[:, 5:7], axis=1)
    if axis == "x":
        cos_rot = trajectory_ref[:, 5] / vel_mag
        sin_rot = -trajectory_ref[:, 6] / vel_mag
    elif axis == "y":
        cos_rot = trajectory_ref[:, 6] / vel_mag
        sin_rot = trajectory_ref[:, 5] / vel_mag

    rotation_matrices[:, 0, 0] = cos_rot
    rotation_matrices[:, 0, 1] = -sin_rot
    rotation_matrices[:, 1, 0] = sin_rot
    rotation_matrices[:, 1, 1] = cos_rot

    transformed_ref = trajectory_ref.copy()
    # translate the position to have it always at 0, 0
    transformed_ref[:, 1:3] -= trajectory_ref[:, 1:3]
    # translate the velocities
    if nullify_velocities:
        transformed_ref[:, 5:7] -= trajectory_ref[:, 5:7]
    # rotate the reference velocity
    transformed_ref[:, 5:7] = np.diagonal(
        np.dot(rotation_matrices, transformed_ref[:, 5:7].T), axis1=0, axis2=2
    ).T

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
        if nullify_velocities:
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
            [
                traj_ped,
                traj_ped_A,
                traj_ped_B,
            ] = compute_simultaneous_observations(
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
            [
                traj_ped,
                traj_ped_A,
                traj_ped_B,
            ] = compute_simultaneous_observations(
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


def compute_length(
    trajectory: np.ndarray,
) -> float:
    """Compute the length of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory

    Returns
    -------
    float
        The length of the trajectory
    """
    list_length = []
    for i in range(len(trajectory) - 1):
        diff = trajectory[i + 1, 1:3] - trajectory[i, 1:3]
        list_length += [np.sqrt(np.sum(diff**2))]
    return np.sum(list_length)


def fit_spline(trajectory: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """Fit a spline to the trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_points : int, optional
        The number of points to use for the spline, by default 1000

    Returns
    -------
    np.ndarray
        The trajectory interpolated with a spline
    """
    cs = CubicSpline(trajectory[:, 0], trajectory[:, 1:3])
    times = np.linspace(trajectory[0, 0], trajectory[-1, 0], n_points)
    interpolation = cs(times)
    return np.hstack((times[:, None], interpolation))


def smooth_with_window_average(
    trajectory: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """Smooth a trajectory using a rolling average

    Parameters
    ----------
    trajetory : np.ndarray
        A trajectory
    window_size : int, optional
        The size of the window to use for the rolling average, by default 10

    Returns
    -------
    np.ndarray
        The smoothed trajectory
    """
    smoothed_trajectory = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(trajectory), i + window_size // 2)
        smoothed_trajectory[i, 1:3] = np.mean(
            trajectory[window_start:window_end, 1:3], axis=0
        )
        smoothed_trajectory[i, 0] = trajectory[i, 0]
        smoothed_trajectory[i, 3:] = trajectory[i, 3:]
    return smoothed_trajectory


def smooth_trajectory_with_window_average(
    trajectory: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """Smooth a trajectory using a rolling average

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    window_size : int, optional
        The size of the window to use for the rolling average, by default 10

    Returns
    -------
    np.ndarray
        The smoothed trajectory
    """
    smoothed_trajectory = np.copy(trajectory)
    for i in range(len(trajectory)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(trajectory), i + window_size // 2)
        smoothed_trajectory[i, 1:3] = np.mean(
            trajectory[window_start:window_end, 1:3], axis=0
        )
        smoothed_trajectory[i, 5:7] = np.mean(
            trajectory[window_start:window_end, 5:7], axis=0
        )
    return smoothed_trajectory


def smooth_trajectory_savitzy_golay(
    trajectory: np.ndarray, window_size: int = 10, order: int = 2
) -> np.ndarray:
    """Smooth a trajectory using a Savitzy-Golay filter

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    window_size : int, optional
        The size of the window to use for the rolling average, by default 10
    order : int, optional
        The order of the polynomial to use, by default 2

    Returns
    -------
    np.ndarray
        The smoothed trajectory
    """
    smoothed_trajectory = np.copy(trajectory)
    smoothed_trajectory[:, 1:3] = savgol_filter(
        trajectory[:, 1:3], window_size, order, axis=0
    )
    smoothed_trajectory[:, 5:7] = savgol_filter(
        trajectory[:, 1:3],
        window_size,
        order,
        axis=0,
        deriv=1,
        delta=np.mean(np.diff(smoothed_trajectory[:, 0])),
    )

    # t = smoothed_trajectory[:, 0]
    # dt = np.diff(t)
    # dp = np.diff(smoothed_trajectory[:, 1:3], axis=0)
    # v = dp / dt[:, None]
    # v = np.concatenate([v, [v[-1]]])
    # smoothed_trajectory[:, 5:7] = v

    return smoothed_trajectory


def smooth_trajectory_gaussian(
    trajectory: np.ndarray, window_size: int = 10, sigma: float = 1
) -> np.ndarray:
    """Smooth a trajectory using a Gaussian filter

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    window_size : int, optional
        The size of the window to use for the rolling average, by default 10
    sigma : float, optional
        The standard deviation of the Gaussian filter, by default 1

    Returns
    -------
    np.ndarray
        The smoothed trajectory
    """
    smoothed_trajectory = np.copy(trajectory)
    smoothed_trajectory[:, 1] = gaussian_filter1d(
        trajectory[:, 1], sigma=sigma, mode="nearest"
    )
    smoothed_trajectory[:, 2] = gaussian_filter1d(
        trajectory[:, 2], sigma=sigma, mode="nearest"
    )

    return smoothed_trajectory


def compute_lateral_distance_obstacle(
    trajectory: np.ndarray, obstacle_position: np.ndarray, n_points: int
) -> np.ndarray:
    dp = trajectory[1:, 1:3] - trajectory[:-1, 1:3]
    mean_dp = np.mean(dp[:n_points], axis=0)
    start_direction = mean_dp / np.linalg.norm(mean_dp)

    # compute the distance between the line
    # guided by the start direction and the obstacle position
    distance = np.abs(cross(start_direction, obstacle_position - trajectory[0, 1:3]))

    # point at which the obstacle is the closest
    # d_to_point = (
    #     np.linalg.norm(obstacle_position - trajectory[0, 1:3]) ** 2 - distance**2
    # ) ** 0.5
    # closest_point = trajectory[0, 1:3] + d_to_point * start_direction

    # plt.scatter(trajectory[:, 1], trajectory[:, 2])
    # plt.scatter(obstacle_position[0], obstacle_position[1])
    # plt.plot(
    #     [trajectory[0, 1], closest_point[0]],
    #     [trajectory[0, 2], closest_point[1]],
    # )
    # plt.axis("equal")
    # plt.show()

    return distance


def compute_stride_frequency_from_residual(
    gait_residual: np.ndarray,
    sampling_time: float = 0.03,
    min_f: float = 0,
    max_f: float = 4,
    n_fft: int = 10000,
    power_threshold: float = 1e-4,
    save_plot: bool = False,
    file_path: str = None,
) -> tuple[float | None, float | None]:
    """
    Compute the stride frequency of a trajectory

    Parameters
    ----------
    gait_residual : np.ndarray
        The gait residual
    sampling_time : float, optional
        The sampling time, by default 0.03
    min_f : float, optional
        The minimum frequency to consider, by default 0
    max_f : float, optional
        The maximum frequency to consider, by default 4
    n_fft : int, optional
        The number of points to use for the FFT, by default 10000
    power_threshold : float, optional
        The minimum power to consider a frequency as valid, by default 1e-4

    Returns
    -------
    float
        The stride frequency or None if no frequency
    float
        The average swaying or None if no frequency
    """
    # compute the periodogram
    sampling_freq = 1 / sampling_time
    f, P = periodogram(gait_residual, sampling_freq, scaling="spectrum", nfft=n_fft)

    # only keep the frequencies between min_f and max_f
    # and the power above the threshold
    mask = (f > min_f) & (f < max_f) & (P > power_threshold)
    candidate_frequencies = f[mask]
    candidate_powers = P[mask]

    if len(candidate_frequencies) == 0:
        return None, None

    # find the frequency with the highest power
    stride_frequency = candidate_frequencies[np.argmax(candidate_powers)]

    # compute the swaying
    stride_t = 1 / stride_frequency
    min_delta = 0.75 * stride_t / 2
    min_delta_points = int(min_delta / sampling_time)
    peaks_idx = find_peaks(
        np.abs(gait_residual), height=0.015, distance=min_delta_points
    )[0]
    # plt.plot(np.abs(gait_residual))
    # plt.scatter(peaks_idx, np.abs(gait_residual)[peaks_idx])
    # plt.show()

    average_swaying = np.mean(np.abs(gait_residual)[peaks_idx])

    if save_plot:
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax.plot(f, P)
        ax.vlines(stride_frequency, 0, np.max(candidate_powers), color="red")
        ax.text(
            stride_frequency + 0.1, np.max(candidate_powers), f"{stride_frequency:.2f}"
        )

        # vertical lines for the min and max frequency
        top_bar = np.max(candidate_powers) * 1.1
        ax.vlines(min_f, 0, top_bar, color="green")
        ax.vlines(max_f, 0, top_bar, color="green")
        # color between the min and max frequency
        ax.fill_between([min_f, max_f], 0, top_bar, color="green", alpha=0.2)

        # horizontal line for the power threshold
        ax.hlines(power_threshold, 0, 4, color="green")

        ax.set_xlim([0, 4])
        ax.set_ylim([0, top_bar])
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (m²)")

        plt.savefig(file_path)
        plt.close()
        # plt.show()

    return stride_frequency, average_swaying


def compute_stride_frequency(
    trajectory: np.ndarray,
    sampling_time: float = 0.03,
    window_duration: float = 2,
    min_f: float = 0,
    max_f: float = 4,
    n_fft: int = 10000,
    power_threshold: float = 0.002,
    save_plot: bool = False,
    file_path: str = None,
) -> tuple[float | None, float | None]:
    """Compute the stride frequency of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    sampling_time : float, optional
        The sampling time, by default 0.03
    window_duration : float, optional
        The duration of the window to use for smoothing, by default 2
    min_f : float, optional
        The minimum frequency to consider, by default 0
    max_f : float, optional
        The maximum frequency to consider, by default 4
    n_fft : int, optional
        The number of points to use for the FFT, by default 10000
    power_threshold : float, optional
        The minimum power to consider a frequency as valid, by default 0.5

    Returns
    -------
    float
        The stride frequency or None if no frequency is found
    """
    gait_residual = compute_gait_residual(trajectory, window_duration=window_duration)

    if gait_residual is None:
        return None, None

    stride_frequency, average_swaying = compute_stride_frequency_from_residual(
        gait_residual,
        sampling_time=sampling_time,
        min_f=min_f,
        max_f=max_f,
        n_fft=n_fft,
        power_threshold=power_threshold,
        save_plot=save_plot,
        file_path=file_path,
    )

    return stride_frequency, average_swaying


def compute_stride_parameters(
    trajectory,
    sampling_time: float = 0.03,
    window_duration: float = 2,
    min_f: float = 0,
    max_f: float = 4,
    n_fft: int = 10000,
    power_threshold: float = 1e-4,
) -> tuple[float | None, float | None, float | None]:
    """Compute the stride frequency of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    sampling_time : float, optional
        The sampling time, by default 0.03
    window_duration : float, optional
        The duration of the window to use for smoothing, by default 2
    min_f : float, optional
        The minimum frequency to consider, by default 0
    max_f : float, optional
        The maximum frequency to consider, by default 4
    n_fft : int, optional
        The number of points to use for the FFT, by default 10000
    power_threshold : float, optional
        The minimum power to consider a frequency as valid, by default 1e-4

    Returns
    -------
    float
        The stride frequency or None if no frequency is found
    float
        The average swaying or None if no frequency is found
    float
        The average stride length or None if no frequency is found
    """

    gait_residual = compute_gait_residual(trajectory, window_duration=window_duration)

    if gait_residual is None:
        return None, None, None

    sampling_freq = 1 / sampling_time
    f, P = periodogram(gait_residual, sampling_freq, scaling="spectrum", nfft=n_fft)

    # only keep the frequencies between min_f and max_f
    # and the power above the threshold
    mask = (f > min_f) & (f < max_f) & (P > power_threshold)
    candidate_frequencies = f[mask]
    candidate_powers = P[mask]

    if len(candidate_frequencies) == 0:
        return None, None, None

    # find the frequency with the highest power
    stride_frequency = candidate_frequencies[np.argmax(candidate_powers)]

    # compute the swaying
    step_t = 1 / (stride_frequency * 2)
    min_delta = 0.5 * step_t
    min_delta_points = int(min_delta / sampling_time)
    peaks_idx = find_peaks(
        np.abs(gait_residual), height=0.015, distance=min_delta_points
    )[0]

    average_swaying = np.mean(np.abs(gait_residual)[peaks_idx])

    fitted_traj = smooth_fitting(trajectory, window_duration=window_duration)

    stride_positions = fitted_traj[peaks_idx, 1:3]

    # plt.plot(trajectory[:, 1], trajectory[:, 2])
    # plt.scatter(stride_positions[:, 0], stride_positions[:, 1])
    # plt.show()

    stride_lengths = np.linalg.norm(np.diff(stride_positions, axis=0), axis=1) * 2
    average_stride_length = np.mean(stride_lengths)

    return stride_frequency, average_swaying, average_stride_length


def smooth_fitting(
    trajectory: np.ndarray, window_duration: float = 2
) -> np.ndarray | None:
    """Smooth a trajectory and compute the velocities

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory

    Returns
    -------
    np.ndarray | None
        The smoothed trajectory
    """
    dt = np.diff(trajectory[:, 0])
    sampling_time = np.mean(dt)
    window = int(window_duration / 0.03)
    if window > len(trajectory):
        return None
    smoothed_trajectory = smooth_trajectory_savitzy_golay(trajectory, window)

    # compute velocities
    t = smoothed_trajectory[:, 0]
    dt = np.diff(t)
    dp = np.diff(smoothed_trajectory[:, 1:3], axis=0) / 1000
    v = dp / dt[:, None]
    v = np.concatenate([v, [v[-1]]])
    smoothed_trajectory[:, 5:7] = v

    return smoothed_trajectory


def compute_gait_residual(
    trajectory: np.ndarray, window_duration: float = 2
) -> np.ndarray | None:
    fitted_traj = smooth_fitting(trajectory, window_duration=window_duration)
    if fitted_traj is None:
        return None

    # find the signed distance to the fitted trajectory
    fitted_position = fitted_traj[:, 1:3]

    step_vectors = np.diff(fitted_position, axis=0)
    step_vectors /= np.linalg.norm(step_vectors, axis=1)[:, None]
    to_fitted = (trajectory[:, 1:3] - fitted_position)[:-1]
    distance_to_fitted = np.cross(step_vectors, to_fitted) / 1000

    return distance_to_fitted


def compute_gsi(
    residual_A: np.ndarray, residual_B: np.ndarray, n_bins: int = 32
) -> float:
    """Compute the gait symmetry index between two trajectories

    Parameters
    ----------
    residual_A : np.ndarray
        The residual of trajectory A
    residual_B : np.ndarray
        The residual of trajectory B
    n_bins : int, optional
        The number of bins to use for the histogram, by default 32

    Returns
    -------
    float
        The gait synchronisation index
    """
    hilbert_A = hilbert(residual_A)
    hilbert_B = hilbert(residual_B)

    phase_A = np.angle(hilbert_A)  # type: ignore
    phase_B = np.angle(hilbert_B)  # type: ignore

    phase_A = np.unwrap(phase_A)
    phase_B = np.unwrap(phase_B)

    phase_diff = phase_A - phase_B

    # wrap the phase difference between -pi and pi
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

    # phase_diff /= 2 * np.pi
    # phase_diff = phase_diff % 1

    hist, _ = np.histogram(phase_diff, bins=n_bins, range=(-np.pi, np.pi))

    ent = entropy(hist)
    gsi = (np.log(n_bins) - ent) / np.log(n_bins)

    return gsi


def compute_coherence(
    residual_A: np.ndarray,
    residual_B: np.ndarray,
    sampling_time: float = 0.03,
    min_freq: float = 0.2,
    max_freq: float = 2.0,
) -> float:
    wct_v, _, coi, freq, _ = wct(residual_A, residual_B, sampling_time, sig=False)

    l = 4 * np.pi / (6 + np.sqrt(2 + 6**2))
    coi_freq = 1 / (l * coi)

    in_coi = freq[:, None] >= coi_freq
    wct_v[~in_coi] = np.nan

    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # t = np.arange(len(residual_A)) * sampling_time

    # ax.pcolormesh(t, freq, np.abs(wct_v), cmap="jet", rasterized=True, linewidth=0)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Frequency (Hz)")
    # ax.set_yscale("log", base=2)
    # ax.set_yticks([int(2**i) if int(2**i) == 2**i else 2**i for i in range(-4, 4)])
    # ax.set_ylim([1 / 16, 10])
    # ax.plot(t, coi_freq, color="black", linestyle="--", linewidth=2)
    # ax.fill_between(t, 1 / 16, coi_freq, alpha=0.5, hatch="//", facecolor="white")

    # plt.show()

    idx_min_f = np.argmin(np.abs(freq - min_freq))
    idx_max_f = np.argmin(np.abs(freq - max_freq))

    wct_v = wct_v[idx_max_f:idx_min_f, :]

    mean_coherence = np.nanmean(wct_v)

    return mean_coherence


def compute_cross_wavelet_transform(
    gait_residuals_A,
    gait_residuals_B,
    min_scale: int = 1,
    max_scale: int = 256,
    n_scales=100,
):
    """
    Compute the cross wavelet transform between two gait residuals

    Parameters
    ----------
    gait_residuals_A : np.ndarray
        The gait residual of pedestrian A
    gait_residuals_B : np.ndarray
        The gait residual of pedestrian B
    min_scale : int, optional
        The minimum scale to consider, by default 1
    max_scale : int, optional
        The maximum scale to consider, by default 256
    n_scales : int, optional
        The number of scales to consider, by default 100

    Returns
    -------
    np.ndarray
        The cross wavelet transform
    """
    scales = np.geomspace(min_scale, max_scale, num=n_scales)
    cwtmatr_A, freqs = pywt.cwt(gait_residuals_A, scales, "cmor0.5-2")
    cwtmatr_B, freqs = pywt.cwt(gait_residuals_B, scales, "cmor0.5-2")

    # fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    # ax[0, 0].plot(gait_residuals_A)
    # ax[0, 0].set_title("Gait residual A")
    # ax[0, 1].plot(gait_residuals_B)
    # ax[0, 1].set_title("Gait residual B")

    # ax[1, 0].imshow(np.abs(cwtmatr_A), aspect="auto", cmap="jet")
    # ax[1, 0].set_title("CWT A")
    # ax[1, 1].imshow(np.abs(cwtmatr_B), aspect="auto", cmap="jet")
    # ax[1, 1].set_title("CWT B")

    # plt.show()

    return cwtmatr_A * np.conj(cwtmatr_B), freqs, scales


def compute_gsi_with_cross_wavelet(
    residual_A: np.ndarray,
    residual_B: np.ndarray,
    min_freq: int = 0.2,
    max_freq: int = 2.0,
    n_bins: int = 32,
    min_scale: int = 1,
    max_scale: int = 256,
    n_scales=100,
    sampling_time: float = 0.03,
) -> float:
    """Compute the gait symmetry index between two trajectories using the cross wavelet transform

    Parameters
    ----------
    residual_A : np.ndarray
        The residual of trajectory A
    residual_B : np.ndarray
        The residual of trajectory B

    Returns
    -------
    float
        The gait synchronisation index
    """

    # plt.plot(residual_A)
    # plt.plot(residual_B)
    # plt.show()

    cwt, freq, scales = compute_cross_wavelet_transform(
        residual_A,
        residual_B,
        min_scale=min_scale,
        max_scale=max_scale,
        n_scales=n_scales,
    )
    # try:
    #     # print(len(residual_A), len(residual_B))
    #     cwt, _, freq, _ = xwt(residual_A, residual_B, sampling_time)
    # except Warning as e:
    #     return None

    phase = np.angle(cwt)

    freq_id_min = np.argmin(np.abs(freq - min_freq))
    freq_id_max = np.argmin(np.abs(freq - max_freq))

    # phase_min_id = np.argmin(np.abs(scales - min_phase))
    # phase_max_id = np.argmin(np.abs(scales - max_phase))
    # print(phase_min_id, phase_max_id)

    # fig, axes = plt.subplots(2, 1)
    # axes[0].imshow(phase, aspect="auto")

    # axes[1].imshow(phase[phase_min_id:phase_max_id, :], aspect="auto")

    # plt.show()

    # phase_band = np.mean(phase[phase_min_id:phase_max_id, :], axis=0)
    phase_band = np.zeros_like(residual_A)
    for i in range(len(residual_A)):
        phase_band[i] = circmean(
            phase[freq_id_max:freq_id_min, i], low=-np.pi, high=np.pi
        )

    # plt.plot(phase_band)
    # plt.show()

    hist, _ = np.histogram(phase_band, bins=n_bins, range=(-np.pi, np.pi))

    ent = entropy(hist)
    gsi = (np.log(n_bins) - ent) / np.log(n_bins)

    return gsi


def compute_gsi_with_phase_locking(
    residual_A: np.ndarray, residual_B: np.ndarray, n_bins: int = 32
) -> float:
    """Compute the gait symmetry index between two trajectories using the phase locking method

    Parameters
    ----------
    residual_A : np.ndarray
        The residual of trajectory A
    residual_B : np.ndarray
        The residual of trajectory B

    Returns
    -------
    float
        The gait synchronisation index
    """
    hilbert_A = hilbert(residual_A)
    hilbert_B = hilbert(residual_B)

    phase_A = np.angle(hilbert_A)  # type: ignore
    phase_B = np.angle(hilbert_B)  # type: ignore

    phase_A = np.unwrap(phase_A)
    phase_B = np.unwrap(phase_B)

    f1 = compute_stride_frequency_from_residual(residual_A, power_threshold=0)[0]
    f2 = compute_stride_frequency_from_residual(residual_B, power_threshold=0)[0]

    # min_n = 0
    # max_n = 2
    # ns = np.linspace(min_n, max_n, 101)
    # ms = np.linspace(min_n, max_n, 101)

    # syncs = np.zeros((len(ns), len(ms)))

    # for i, n in enumerate(ns):
    #     for j, m in enumerate(ms):
    #         phi_nm = n * phase_A - m * phase_B

    #         phi_nm = (phi_nm + np.pi) % (2 * np.pi) - np.pi

    #         hist, _ = np.histogram(phi_nm, bins=n_bins, range=(-np.pi, np.pi))
    #         ent = entropy(hist)
    #         sync = (np.log(n_bins) - ent) / np.log(n_bins)

    #         syncs[i, j] = sync

    phi_nm = f2 * phase_A - f1 * phase_B
    phi_nm = (phi_nm + np.pi) % (2 * np.pi) - np.pi
    hist_nm, _ = np.histogram(phi_nm, bins=n_bins, range=(-np.pi, np.pi))
    ent_nm = entropy(hist_nm)
    sync_nm = (np.log(n_bins) - ent_nm) / np.log(n_bins)

    # phi_11 = phase_A - phase_B
    # phi_11 = (phi_11 + np.pi) % (2 * np.pi) - np.pi
    # hist_11, _ = np.histogram(phi_11, bins=n_bins, range=(-np.pi, np.pi))
    # ent_11 = entropy(hist_11)
    # sync_11 = (np.log(n_bins) - ent_11) / np.log(n_bins)

    # if sync_nm < sync_11:
    #     print(f"n = {f1}, m = {f2}, sync = {sync_nm}, sync_11 = {sync_11}")
    #     fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    #     im = ax.imshow(
    #         syncs, cmap="viridis", origin="lower", extent=[min_n, max_n, min_n, max_n]
    #     )
    #     cbar = fig.colorbar(im)

    #     ax.scatter(f1, f2, color="red", marker="x", s=100)
    #     ax.scatter(1, 1, color="green", marker="x", s=100)
    #     ax.set_xlabel("n")
    #     ax.set_ylabel("m")
    #     ax.set_title("Phase locking")
    #     plt.show()

    return sync_nm


def compute_average_synchronization(
    traj_A: np.ndarray,
    traj_B: np.ndarray,
    n_bins: int = 32,
    window_duration: int | None = None,
    window_length: int | None = None,
    method: str = "hilbert",
    nm_locking: bool = False,
    min_freq: int = 0.5,
    max_freq: int = 2.0,
    min_scale: int = 1,
    max_scale: int = 256,
    n_scales=100,
    power_threshold: float = 1e-4,
) -> tuple:
    """
    Compute the average gait symmetry index, the average mean relative phase, the average variance of the relative phase

    Parameters
    ----------
    traj_A : np.ndarray
        The trajectory of pedestrian A
    traj_B : np.ndarray
        The trajectory of pedestrian B
    n_bins : int, optional
        The number of bins to use for the histogram, by default 32
    window_duration : int, optional
        The duration of the window used for computing the average gait symmetry index, by default 5 s
    window_length : int, optional
        The length of the window used for computing the average gait symmetry index, by default 5 m
    method : str, optional
        The method to use for computing the average gait symmetry index, by default "hilbert"
    nm_locking : bool, optional
        Whether or not to use n m phase locking, by default False
    min_freq : int, optional
        The minimum frequency, by default 0.5
    max_freq : int, optional
        The maximum frequency, by default 2.0
    min_scale : int, optional
        The minimum scale to consider for the cross wavelet transform, by default 1
    max_scale : int, optional
        The maximum scale to consider for the cross wavelet transform, by default 256
    n_scales : int, optional
        The number of scales to consider for the cross wavelet transform, by default 100
    power_threshold : float, optional
        The minimum power to consider a frequency as valid, by default 1e-4
    Returns
    -------
    tuple | None
        The average gait symmetry index, the average mean relative phase, the average variance of the relative phase

    """

    assert len(traj_A) == len(traj_B)

    assert window_duration is not None or window_length is not None
    assert window_duration is None or window_length is None

    assert method in ["hilbert", "wavelet"]

    gait_residual_A = compute_gait_residual(traj_A)
    gait_residual_B = compute_gait_residual(traj_B)

    if gait_residual_A is None or gait_residual_B is None:
        return None, None, None, None

    if window_duration is not None:
        split_indices = get_pieces_indices_from_time(traj_A, window_duration)
    else:
        split_indices = get_pieces_indices(traj_A, window_length)

    if len(split_indices) == 0:
        return None, None, None, None

    # print(split_indices)
    # sub_trajectories_A = [
    #     traj_A[start:end] for start, end in split_indices if len(traj_A[start:end])
    # ]
    # sub_trajectories_B = [
    #     traj_B[start:end] for start, end in split_indices if len(traj_B[start:end])
    # ]
    # colors = ["blue", "red"] * len(sub_trajectories_A)
    # plot_static_2D_trajectories(sub_trajectories_A + sub_trajectories_B, colors=colors)

    gsis, means_relative_phase, variances_relative_phase = [], [], []
    delta_fs = []
    for start, end in split_indices:
        n_points = end - start
        if n_points < 10:  # not enough points
            continue

        # compute gsi
        if nm_locking:
            gsi = compute_gsi_with_phase_locking(
                gait_residual_A[start:end], gait_residual_B[start:end], n_bins
            )
        else:
            if method == "hilbert":
                gsi = compute_gsi(
                    gait_residual_A[start:end], gait_residual_B[start:end], n_bins
                )
            elif method == "wavelet":
                gsi = compute_gsi_with_cross_wavelet(
                    gait_residual_A[start:end],
                    gait_residual_B[start:end],
                    min_freq=min_freq,
                    max_freq=max_freq,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    n_scales=n_scales,
                )

        if gsi is None:
            continue

        gsis += [gsi]

        # compute mean relative phase
        if method == "hilbert":
            mean_relative_phase, variance_relative_phase = compute_mean_relative_phase(
                gait_residual_A[start:end], gait_residual_B[start:end]
            )
        elif method == "wavelet":
            mean_relative_phase, variance_relative_phase = (
                compute_mean_relative_phase_with_cross_wavelet(
                    gait_residual_A[start:end],
                    gait_residual_B[start:end],
                )
            )
        means_relative_phase += [mean_relative_phase]
        variances_relative_phase += [variance_relative_phase]

        # compute delta_f
        f1 = compute_stride_frequency_from_residual(
            gait_residual_A[start:end],
            power_threshold=power_threshold,
            min_f=min_freq,
            max_f=max_freq,
            # save_plot=True,
            n_fft=1000,
        )[0]
        f2 = compute_stride_frequency_from_residual(
            gait_residual_B[start:end],
            power_threshold=power_threshold,
            min_f=min_freq,
            max_f=max_freq,
            # save_plot=True,
            n_fft=1000,
        )[0]

        if f1 is None or f2 is None:
            delta_f = np.nan
        else:
            delta_f = np.abs(f1 - f2)
            # print(delta_f)
        delta_fs += [delta_f]

    # if gsi > 0.3:
    #     plot_static_2D_trajectories([traj_A[start:end], traj_B[start:end]])

    return (
        np.nanmean(gsis),
        circmean(means_relative_phase, high=np.pi, low=-np.pi),
        np.mean(variances_relative_phase),
        np.mean(delta_fs),
    )


def compute_average_synchronization_from_vel(
    traj_A: np.ndarray,
    traj_B: np.ndarray,
    n_bins: int = 32,
    window_duration: int = None,
    window_length: int = None,
) -> tuple:
    """
    Compute the average gait symmetry index, the average mean relative phase, the average variance of the relative phase

    Parameters
    ----------
    traj_A : np.ndarray
        The trajectory of pedestrian A
    traj_B : np.ndarray
        The trajectory of pedestrian B
    n_bins : int, optional
        The number of bins to use for the histogram, by default 32
    window_duration : int, optional
        The duration of the window used for computing the average gait symmetry index, by default 5 s
    window_length : int, optional
        The length of the window used for computing the average gait symmetry index, by default 5 m
    Returns
    -------
    tuple | None
        The average gait symmetry index, the average mean relative phase, the average variance of the relative phase

    """

    assert len(traj_A) == len(traj_B)

    assert window_duration is not None or window_length is not None
    assert window_duration is None or window_length is None

    vel_A = compute_velocity(traj_A)
    vel_B = compute_velocity(traj_B)

    vel_mag_A = np.linalg.norm(vel_A, axis=1)
    vel_mag_B = np.linalg.norm(vel_B, axis=1)

    vel_mag_A = vel_A - np.mean(vel_A, axis=0)
    vel_mag_B = vel_B - np.mean(vel_B, axis=0)

    # plt.plot(traj_A[:, 0], vel_mag_A, label="A")
    # plt.plot(traj_B[:, 0], vel_mag_B, label="B")
    # plt.legend()
    # plt.show()

    # if gait_residual_A is None or gait_residual_B is None:
    #     return None, None, None

    if window_duration is not None:
        split_indices = get_pieces_indices_from_time(traj_A, window_duration)
    else:
        split_indices = get_pieces_indices(traj_A, window_length)

    if len(split_indices) == 0:
        return None, None, None

    # print(split_indices)
    # sub_trajectories_A = [
    #     traj_A[start:end] for start, end in split_indices if len(traj_A[start:end])
    # ]
    # sub_trajectories_B = [
    #     traj_B[start:end] for start, end in split_indices if len(traj_B[start:end])
    # ]
    # colors = ["blue", "red"] * len(sub_trajectories_A)
    # plot_static_2D_trajectories(sub_trajectories_A + sub_trajectories_B, colors=colors)

    gsis, means_relative_phase, variances_relative_phase = [], [], []
    for start, end in split_indices:
        gsi = compute_gsi(vel_mag_A[start:end], vel_mag_B[start:end], n_bins)
        if gsi is None:
            continue

        gsis += [gsi]
        mean_relative_phase, variance_relative_phase = compute_mean_relative_phase(
            vel_mag_A[start:end], vel_mag_B[start:end]
        )
        means_relative_phase += [mean_relative_phase]
        variances_relative_phase += [variance_relative_phase]

    # if gsi > 0.3:
    #     plot_static_2D_trajectories([traj_A[start:end], traj_B[start:end]])

    return (
        np.mean(gsis),
        circmean(means_relative_phase, high=np.pi, low=-np.pi),
        np.mean(variances_relative_phase),
    )


def compute_mean_relative_phase(
    gait_residual_A: np.ndarray, gait_residual_B: np.ndarray
):
    """Compute the mean relative phase between two gait residuals

    Parameters
    ----------
    gait_residual_A : np.ndarray
        The gait residual of trajectory A
    gait_residual_B : np.ndarray
        The gait residual of trajectory B

    Returns
    -------
    mean_dphi : float
        The mean relative phase
    variance_dphi : float
        The variance of the relative phase

    """
    hilbert_A = hilbert(gait_residual_A)
    hilbert_B = hilbert(gait_residual_B)

    phase_A = np.angle(hilbert_A)  # type: ignore
    phase_B = np.angle(hilbert_B)  # type: ignore
    # get phase between 0 and 2pi
    phase_A[phase_A < 0] += 2 * np.pi
    phase_B[phase_B < 0] += 2 * np.pi

    dphi = phase_A - phase_B
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi

    mean_dphi = circmean(dphi, high=np.pi, low=-np.pi)
    variance_dphi = circvar(dphi, high=np.pi, low=-np.pi)

    return mean_dphi, variance_dphi


def compute_mean_relative_phase_with_cross_wavelet(
    gait_residual_A: np.ndarray,
    gait_residual_B: np.ndarray,
    min_phase: int = 60,
    max_phase: int = 85,
    min_scale: int = 1,
    max_scale: int = 256,
    n_scales=100,
):
    """Compute the mean relative phase between two gait residuals using the cross wavelet transform

    Parameters
    ----------
    gait_residual_A : np.ndarray
        The gait residual of trajectory A
    gait_residual_B : np.ndarray
        The gait residual of trajectory B
    min_phase : int, optional
        The minimum phase to consider, by default 60
    max_phase : int, optional
        The maximum phase to consider, by default 85
    min_scale : int, optional
        The minimum scale to consider, by default 1
    max_scale : int, optional
        The maximum scale to consider, by default 256
    n_scales : int, optional
        The number of scales to consider, by default 100


    Returns
    -------
    tuple
        The mean relative phase, the variance of the relative phase

    """

    cwt, _, scales = compute_cross_wavelet_transform(
        gait_residual_A,
        gait_residual_B,
        min_scale=min_scale,
        max_scale=max_scale,
        n_scales=n_scales,
    )

    phase = np.angle(cwt)

    phase_min_id = np.argmin(np.abs(scales - min_phase))
    phase_max_id = np.argmin(np.abs(scales - max_phase))

    phase_band = np.mean(phase[phase_min_id:phase_max_id, :], axis=0)

    mean_dphi = circmean(phase_band, high=np.pi, low=-np.pi)
    variance_dphi = circvar(phase_band, high=np.pi, low=-np.pi)

    return mean_dphi, variance_dphi


def compute_velocity(trajectory: np.ndarray) -> np.ndarray:
    """Compute the velocity of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory

    Returns
    -------
    np.ndarray
        The velocity of the trajectory
    """
    dt = np.diff(trajectory[:, 0])
    dp = np.diff(trajectory[:, 1:3], axis=0)
    v = dp / dt[:, None]
    # print(v)
    v = np.concatenate([v, [v[-1]]])
    return v


def compute_average_velocity(trajectory: np.ndarray) -> float:
    """Compute the average velocity of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory

    Returns
    -------
    float
        The average velocity of the trajectory
    """
    v = compute_velocity(trajectory)
    return np.mean(np.linalg.norm(v, axis=1))


def compute_integral_deviation(
    trajectory: np.ndarray, n_average=66, normalize=True
) -> float:
    velocities = compute_velocity(trajectory)

    start_point = trajectory[0, 1:3]
    middle_points = trajectory[1:, 1:3]

    start_vel = np.nanmean(velocities[:n_average], axis=0)
    start_vel /= np.linalg.norm(start_vel)

    distances_to_straight_line = cross(start_vel, middle_points - start_point)
    distances_to_origin = np.linalg.norm(middle_points - start_point, axis=1)
    distance_to_projection = np.sqrt(
        distances_to_origin**2 - distances_to_straight_line**2
    )
    # projections = start_point + distance_to_projection[
    #     :, None
    # ] * start_vel / np.linalg.norm(start_vel)

    # cum_distances_to_projection = np.cumsum(distance_to_projection)
    integral = np.abs(
        trapz(distances_to_straight_line / 1000, distance_to_projection / 1000)
    )

    # plt.plot(distance_to_projection, distances_to_straight_line)
    # plt.show()

    if normalize:
        net = compute_net_displacement(trajectory[:, 1:3]) / 1000
        integral /= net

    # plt.scatter(trajectory[:, 1], trajectory[:, 2], s=1)
    # plt.plot(
    #     [start_point[0], start_point[0] + 4000 * start_vel[0]],
    #     [start_point[1], start_point[1] + 4000 * start_vel[1]],
    # )
    # plt.scatter(projections[:, 0], projections[:, 1], s=1)
    # for i in range(len(projections)):
    #     plt.plot(
    #         [middle_points[i, 0], projections[i, 0]],
    #         [middle_points[i, 1], projections[i, 1]],
    #         color="red",
    #     )
    # plt.axis("equal")
    # plt.show()

    return integral


def compute_integral_cumulative_turning_angle(
    trajectory: np.ndarray,
    step_length: None | float = None,
    rediscretize=True,
    normalize=True,
) -> float:
    position = trajectory[:, 1:3]
    if rediscretize and step_length is not None:
        position = rediscretize_position_v2(position, step_length=step_length)
        if len(position) <= 1:
            return np.nan
    turning_angles = compute_turning_angles(position)

    cum_sum = np.cumsum(turning_angles)

    return np.mean(np.abs(cum_sum))

    # t = trajectory[1:-1, 0]

    # integral = np.abs(np.trapz(cum_sum, t))

    # if normalize:
    #     net = compute_net_displacement(trajectory[:, 1:3]) / 1000
    #     integral /= net

    # return integral


def compute_dynamic_time_warping_distance(trajectory_A, trajectory_B) -> float:
    """Compute the dynamic time warping distance between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory

    Returns
    -------
    float
        The dynamic time warping distance between the two trajectories
    """
    # compute the dynamic time warping
    n_points_A = len(trajectory_A)
    n_points_B = len(trajectory_B)

    dtw = np.full((n_points_A, n_points_B), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n_points_A):
        for j in range(1, n_points_B):
            cost = np.linalg.norm(trajectory_A[i, 1:3] - trajectory_B[j, 1:3])
            dtw[i, j] = cost + np.min([dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]])

    return dtw[-1, -1] / max([n_points_A, n_points_B]) / 1000


def compute_dynamic_time_warping_deviation(
    trajectory: np.ndarray, n_average=66
) -> float:
    """
    Compute the time warping deviation of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The time warping deviation of the trajectory
    """

    n_points = len(trajectory)
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_v * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    # plt.scatter(trajectory[:, 1], trajectory[:, 2], s=1)
    # plt.scatter(straight_line_trajectory[:, 1], straight_line_trajectory[:, 2], s=1)
    # plt.show()

    dtw = compute_dynamic_time_warping_distance(trajectory, straight_line_trajectory)
    return dtw


def compute_lcss(trajectory_A, trajectory_B, eps=50):
    """Compute the longest common subsequence between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory
    eps : float, optional
        The epsilon parameter to use, by default 0.1

    Returns
    -------
    float
        The longest common subsequence between the two trajectories
    """
    n_points_A = len(trajectory_A)
    n_points_B = len(trajectory_B)

    lcss = np.zeros((n_points_A, n_points_B))

    for i in range(1, n_points_A):
        for j in range(1, n_points_B):
            d = np.linalg.norm(trajectory_A[i, 1:3] - trajectory_B[j, 1:3]).astype(
                float
            )
            if d < eps:
                lcss[i, j] = lcss[i - 1, j - 1] + 1
            else:
                lcss[i, j] = np.max([lcss[i - 1, j], lcss[i, j - 1]])
    return lcss[-1, -1]


def compute_lcss_distance(trajectory_A, trajectory_B, eps=50, normalize=False):
    """Compute the lcss distance between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory
    eps : float, optional
        The epsilon parameter to use, by default 0.1
    normalize : bool, optional
        Whether or not to normalize the distance, by default False

    Returns
    -------
    float
        The lcss distance between the two trajectories
    """
    lcss = compute_lcss(trajectory_A, trajectory_B, eps=eps)
    d = 1 - lcss / min([len(trajectory_A), len(trajectory_B)])
    # if not normalize:
    #     d = len(trajectory_A) + len(trajectory_B) - 2 * lcss
    # else:
    #     d = 1 - lcss / (len(trajectory_A) + len(trajectory_B) - lcss)
    return d


def compute_lcss_deviation(trajectory: np.ndarray, n_average=66, eps=50) -> float:
    """
    Compute the lcss deviation of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The lcss deviation of the trajectory
    """

    n_points = len(trajectory)
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_v * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    lcss = compute_lcss_distance(trajectory, straight_line_trajectory, eps=eps)
    return lcss


def compute_edr_distance(trajectory_A, trajectory_B, eps=50):
    """Compute the edit distance on real sequence between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory
    eps : float, optional
        The epsilon parameter to use, by default 0.1

    Returns
    -------
    float
        The edit distance on real sequence between the two trajectories
    """
    # compute the edit distance on real
    n_points_A = len(trajectory_A)
    n_points_B = len(trajectory_B)

    edr = np.full((n_points_A, n_points_B), np.inf)
    edr[0, 0] = 0

    for i in range(1, n_points_A):
        for j in range(1, n_points_B):
            d = np.linalg.norm(trajectory_A[i, 1:3] - trajectory_B[j, 1:3]).astype(
                float
            )
            if d < eps:
                c = 0
            else:
                c = 1
            edr[i, j] = np.min(
                [
                    edr[i - 1, j] + 1,
                    edr[i, j - 1] + 1,
                    edr[i - 1, j - 1] + c,
                ]
            )
    return edr[-1, -1]


def compute_edr_deviation(trajectory: np.ndarray, n_average=66, eps=50) -> float:
    """
    Compute the edr deviation of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The edr deviation of the trajectory
    """

    n_points = len(trajectory)
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_v * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    edr = compute_edr_distance(trajectory, straight_line_trajectory, eps=eps)
    return edr


def compute_euclidean_distance(trajectory_A, trajectory_B):
    """Compute the euclidean distance between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory

    Returns
    -------
    float
        The euclidean distance between the two trajectories
    """
    assert len(trajectory_A) == len(trajectory_B)
    return (
        np.sum(np.linalg.norm(trajectory_A[:, 1:3] - trajectory_B[:, 1:3], axis=1))
        / 1000
        / len(trajectory_A)
    )


def compute_euclidean_deviation(trajectory: np.ndarray, n_average=66) -> float:
    """
    Compute the euclidean deviation of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The euclidean deviation of the trajectory
    """

    n_points = len(trajectory)
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_v * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    ed = compute_euclidean_distance(trajectory, straight_line_trajectory)
    return ed


def compute_erp_distance(trajectory_A, trajectory_B, g=(0, 0)):
    """Compute the edit distance with real penalty between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory
    g : tuple, optional
        The gap points, by default (0,0)
    Returns
    -------
    float
        The edit distance with real penalty between the two trajectories
    """
    # compute the edit distance with real penalty
    n_points_A = len(trajectory_A)
    n_points_B = len(trajectory_B)

    dist_g_A = np.linalg.norm(trajectory_A[:, 1:3] - g, axis=1)
    dist_g_B = np.linalg.norm(trajectory_B[:, 1:3] - g, axis=1)

    erp = np.full((n_points_A, n_points_B), np.inf)
    erp[0, 0] = 0

    for i in range(1, n_points_A):
        for j in range(1, n_points_B):
            d = np.linalg.norm(trajectory_A[i, 1:3] - trajectory_B[j, 1:3]).astype(
                float
            )
            erp[i, j] = np.min(
                [
                    erp[i - 1, j] + dist_g_A[i],
                    erp[i, j - 1] + dist_g_B[j],
                    erp[i - 1, j - 1] + d,
                ]
            )
    return erp[-1, -1] / 1000


def compute_erp_deviation(trajectory: np.ndarray, n_average=66) -> float:
    """
    Compute the erp deviation of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The erp deviation of the trajectory
    """

    n_points = len(trajectory)
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_v * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    g = (0, 0)

    erp = compute_erp_distance(trajectory, straight_line_trajectory, g=g)
    return erp


def compute_discrete_frechet_distance(trajectory_A, trajectory_B):
    """
    Compute the discrete Frechet distance between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory

    Returns
    -------
    float
        The discrete Frechet distance between the two trajectories
    """

    n_points_A = len(trajectory_A)
    n_points_B = len(trajectory_B)

    df = distance.cdist(trajectory_A[:, 1:3], trajectory_B[:, 1:3])

    for i in range(1, n_points_A):
        df[i, 0] = np.max([df[i - 1, 0], df[i, 0]])
    for j in range(1, n_points_B):
        df[0, j] = np.max([df[0, j - 1], df[0, j]])

    for i in range(1, n_points_A):
        for j in range(1, n_points_B):
            df[i, j] = np.max(
                [np.min([df[i - 1, j], df[i, j - 1], df[i - 1, j - 1]]), df[i, j]]
            )

    return df[-1, -1] / 1000


def compute_simultaneous_frechet_distance(trajectory_A, trajectory_B):
    """
    Compute the simultaneous Frechet distance between two trajectories

    Parameters
    ----------
    trajectory_A : np.ndarray
        A trajectory
    trajectory_B : np.ndarray
        A trajectory

    Returns
    -------
    float
        The simultaneous Frechet distance between the two trajectories
    """

    distances = np.linalg.norm(trajectory_A[:, 1:3] - trajectory_B[:, 1:3], axis=1)
    return np.max(distances) / 1000


def compute_simultaneous_frechet_deviation(
    trajectory: np.ndarray, n_average=66
) -> float:
    n_points = len(trajectory)
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_v * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    df = compute_simultaneous_frechet_distance(trajectory, straight_line_trajectory)
    return df


def compute_discrete_frechet_deviation(trajectory: np.ndarray, n_average=66) -> float:
    """
    Compute the discrete Frechet deviation of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The discrete Frechet deviation of the trajectory
    """

    n_points = len(trajectory)
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_v * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    df = compute_discrete_frechet_distance(trajectory, straight_line_trajectory)
    return df


def compute_steps(trajectory, n_average=66):
    """
    Find the positions of the step of a trajectory (i.e. the positions where
    the angle of direction changes from negative to positive or vice versa)

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The number of steps of the trajectory
    """

    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)
    start_direction = start_v / np.linalg.norm(start_v)

    pos_diff = np.diff(trajectory[:, 1:3], axis=0)
    pos_diff_dir = pos_diff / np.linalg.norm(pos_diff, axis=1)[:, None]

    sign_det = np.sign(np.cross(start_direction, pos_diff_dir))
    sign_changes = np.diff(sign_det)
    sign_changes = np.concatenate([[-2], sign_changes, [-2]])

    steps_positions = trajectory[:, 1:3][sign_changes != 0]

    # start_direction_angle = np.arctan2(start_direction[1], start_direction[0])

    # pos_diff = np.diff(trajectory[:, 1:3], axis=0)
    # pos_diff_dir = pos_diff / np.linalg.norm(pos_diff, axis=1)[:, None]
    # # compute signed angle between start direction and step direction
    # pos_diff_angles = np.arctan2(pos_diff_dir[:, 1], pos_diff_dir[:, 0])
    # step_angles_to_desired = pos_diff_angles - start_direction_angle
    # # print(step_angles_to_desired)
    # sign_changes = np.diff(np.sign(step_angles_to_desired))
    # sign_changes = np.concatenate([[-2], sign_changes, [-2]])

    # steps_positions = trajectory[:, 1:3][sign_changes != 0]

    # # sign_changes = np.diff(np.sign(step_angles))
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(trajectory[:, 1], trajectory[:, 2])
    # ax[0].set_aspect("equal")
    # ax[1].plot(np.rad2deg(pos_diff_angles))
    # ax[1].plot(np.rad2deg(step_angles_to_desired))
    # ax[1].plot(np.rad2deg(start_direction_angle))
    # # ax[1].plot(sign_changes)
    # # ax[2].plot(sign_changes)
    # plt.show()

    return steps_positions


def compute_suddenness_turn(trajectory, n_average=66) -> float:
    """
    Compute the suddenness of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The suddenness of the trajectory
    """

    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)
    start_direction = start_v / np.linalg.norm(start_v)
    start_angle = np.arctan2(start_direction[1], start_direction[0])
    # normalize between 0 and 2pi
    if start_angle < 0:
        start_angle += 2 * np.pi

    steps = np.diff(trajectory[:, 1:3], axis=0)
    step_directions = steps / np.linalg.norm(steps, axis=1)[:, None]
    step_angles = np.arctan2(step_directions[:, 1], step_directions[:, 0])
    step_angles[step_angles < 0] += 2 * np.pi

    step_angles_to_desired = np.abs(step_angles - start_angle)
    step_angles_to_desired[step_angles_to_desired > np.pi] = (
        2 * np.pi - step_angles_to_desired[step_angles_to_desired > np.pi]
    )
    step_angles_to_desired = np.rad2deg(step_angles_to_desired)

    vel_mag = np.linalg.norm(velocities, axis=1)
    velocity_diff = np.diff(vel_mag) * 100  # cm/s

    suddenness = np.abs(step_angles_to_desired * velocity_diff)

    # fig, ax = plt.subplots(4, 1)
    # ax[0].plot(trajectory[:, 1], trajectory[:, 2], '-o')
    # ax[0].set_aspect("equal")
    # ax[1].plot(step_angles)
    # ax[2].plot(velocity_diff)
    # ax[3].plot(suddenness)
    # plt.show()

    return suddenness


def compute_turn_intensity(trajectory, n_average=66) -> float:
    """
    Compute the turn intensity of a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        A trajectory
    n_average : int, optional
        The number of points to use for the average velocity, by default 66

    Returns
    -------
    float
        The turn intensity of the trajectory
    """
    velocities = compute_velocity(trajectory)
    start_v = np.nanmean(velocities[:n_average], axis=0)
    start_direction = start_v / np.linalg.norm(start_v)
    start_angle = np.arctan2(start_direction[1], start_direction[0])
    # normalize between 0 and 2pi
    if start_angle < 0:
        start_angle += 2 * np.pi

    steps_positions = compute_steps(trajectory, n_average=n_average)
    steps = np.diff(steps_positions, axis=0)
    step_directions = steps / np.linalg.norm(steps, axis=1)[:, None]
    step_angles = np.arctan2(step_directions[:, 1], step_directions[:, 0])
    step_angles[step_angles < 0] += 2 * np.pi

    step_angles_to_desired = np.abs(step_angles - start_angle)
    step_angles_to_desired[step_angles_to_desired > np.pi] = (
        2 * np.pi - step_angles_to_desired[step_angles_to_desired > np.pi]
    )

    step_deviation = np.linalg.norm(steps, axis=1) * np.sin(step_angles_to_desired) / 10
    step_angles_to_desired = np.rad2deg(step_angles_to_desired)

    turn_intensity = step_angles_to_desired * step_deviation
    # print(turn_intensity)

    # fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    # ax[0].plot(trajectory[:, 1], trajectory[:, 2])
    # ax[0].plot(steps_positions[:, 0], steps_positions[:, 1], "-o")
    # # vector for the start direction
    # ax[0].quiver(
    #     trajectory[0, 1],
    #     trajectory[0, 2],
    #     start_direction[0],
    #     start_direction[1],
    #     scale=10,
    # )
    # ax[0].axis("equal")

    # ax[1].plot(step_angles_to_desired)
    # ax[2].plot(step_deviation)
    # ax[3].plot(turn_intensity)

    # plt.show()

    return turn_intensity


def compute_rqa(
    gait_residual_A,
    gait_residual_B,
    embedding_dimension=4,
    time_delay=7,
    epsilon=0.07,
    theiler_corrector=0,
):
    """Compute the recurrence quantification analysis between two gait residuals

    Parameters
    ----------
    gait_residual_A : np.ndarray
        The gait residual of trajectory A
    gait_residual_B : np.ndarray
        The gait residual of trajectory B
    embedding_dimension : int, optional
        The embedding dimension, by default 4
    time_delay : int, optional
        The time delay, by default 7
    epsilon : float, optional
        The epsilon parameter, by default 0.07
    theiler_corrector : int, optional
        The theiler corrector, by default 0

    Returns
    -------
    tuple
        The recurrence rate, the determinism, the longest diagonal line
    """

    time_series_A = TimeSeries(
        gait_residual_A, embedding_dimension=embedding_dimension, time_delay=time_delay
    )
    time_series_B = TimeSeries(
        gait_residual_B, embedding_dimension=embedding_dimension, time_delay=time_delay
    )
    time_series = (time_series_A, time_series_B)
    settings = Settings(
        time_series,
        analysis_type=Cross,  # type: ignore
        neighbourhood=FixedRadius(epsilon),
        similarity_measure=EuclideanMetric,
        theiler_corrector=theiler_corrector,
    )
    computation = RQAComputation.create(settings)
    result = computation.run()
    rec = result.recurrence_rate
    det = result.determinism
    maxline = result.longest_diagonal_line

    return rec, det, maxline


def compute_lyapunov_exponent(
    p,
    n_iterations=1000,
    n_neighbors=10,
    n_points=500,
    eps=0.03,
    theiler_window=400,
    ax=None,
):
    """Compute the maximuam Lyapunov exponent of a trajectory

    Parameters
    ----------
    p : np.ndarray
        The trajectory
    n_iterations : int, optional
        The number of iterations, by default 1000
    n_neighbors : int, optional
        The number of neighbors, by default 10
    n_points : int, optional
        The number of points, by default 500
    eps : float, optional
        The epsilon parameter, by default 0.03
    theiler_window : int, optional
        The theiler window, by default 400
    ax : matplotlib.pyplot.Axes, optional
        The axis to plot the results, by default None

    Returns
    -------
    float
        The maximal Lyapunov exponent
    """

    max_try = n_points * 10

    distances = []
    j = 0

    n_try = 0

    while j < n_points and n_try < max_try:
        # random starting point in the embedding
        idx_reference_point = np.random.randint(0, p.shape[0])
        point = p[idx_reference_point]
        # find the nearest neighbors
        all_distances = np.linalg.norm(p - point, axis=1)
        all_distances[idx_reference_point] = np.inf  # exclude the reference point
        # id of the close enough neighbors
        neighbors = np.where(all_distances < eps)[0]
        n_try += 1
        # keep only the neighbors that are not too close to the reference point
        neighbors = neighbors[np.abs(neighbors - idx_reference_point) > theiler_window]

        if len(neighbors) < n_neighbors:
            continue
        # compute the average distance at all iterations
        point_distances = np.zeros(n_iterations)
        for k in range(n_iterations):
            reference_trajectory_point = p[(idx_reference_point + k) % p.shape[0]]
            for neighbor in neighbors:
                neighbor_trajectory_point = p[(neighbor + k) % p.shape[0]]
                point_distances[k] += np.abs(
                    reference_trajectory_point[-1] - neighbor_trajectory_point[-1]
                )
        point_distances /= len(neighbors)
        j += 1
        distances.append(point_distances)

    distances = np.array(distances)

    if len(distances) == 0:
        return None

    expansion_rate = np.nanmean(np.log(distances), axis=0)

    # fit a line with method of least squares
    A = np.vstack([np.arange(n_iterations), np.ones(n_iterations)]).T
    m, c = np.linalg.lstsq(A, expansion_rate, rcond=None)[0]

    if ax is not None:
        ax.plot(expansion_rate)
        ax.plot(np.arange(n_iterations), m * np.arange(n_iterations) + c)
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("$\\log(E)$")
        ax.grid(color="gray", linestyle="--", linewidth=0.5)

    return m


def get_box(point, limits, n_dimensions, n_boxes=10):
    """Get the box of a point

    Parameters
    ----------
    point : np.array
        The point
    limits : np.array
        The limits of the space
    n_dimensions : int
        The number of dimensions
    n_boxes : int
        The number of boxes

    Returns
    -------
    np.array
        The box of the point
    """
    box = np.zeros(n_dimensions)
    for i in range(n_dimensions):
        box[i] = np.floor(
            (point[i] - limits[i, 0]) / (limits[i, 1] - limits[i, 0]) * n_boxes
        )
        if box[i] == n_boxes:
            box[i] -= 1
    return box


def check_determinism(p, n_boxes=10, min_val_box=10, ax=None):
    """
    Check the determinism of a trajectory using Kaplan's method

    Parameters
    ----------
    p : np.array
        The embedded trajectory, shape (n_points, n_dimensions)
    n_boxes : int
        The number of boxes
    min_val_box : int
        The minimum number of values in a box that are needed to compute the average box direction

    Returns
    -------
    float
        The average box direction across all boxes,
        a value close to 1 indicates a deterministic trajectory
    """

    n_dimensions = p.shape[1]
    limit = np.zeros((n_dimensions, 2))
    for i in range(n_dimensions):
        limit[i, 0] = np.min(p[:, i])
        limit[i, 1] = np.max(p[:, i])

    first_point = p[0, :]
    current_box = get_box(p[0, :], limit, n_dimensions, n_boxes)

    box_directions = {}

    for j in range(1, len(p)):
        new_box = get_box(p[j, :], limit, n_dimensions, n_boxes)
        if not np.all(new_box == current_box):
            # reached a new box
            direction = (p[j, :] - first_point) / np.linalg.norm(p[j, :] - first_point)
            if tuple(current_box) not in box_directions:
                box_directions[tuple(current_box)] = []
            box_directions[tuple(current_box)].append(direction)

            first_point = p[j, :]  # update the first point
        current_box = new_box

    average_box_directions = []
    for _, directions in box_directions.items():
        if len(directions) >= min_val_box:
            average_box_direction = np.mean(directions, axis=0)
            norm_average_box_direction = np.linalg.norm(average_box_direction)
            average_box_directions.append(norm_average_box_direction)

    if len(average_box_directions) == 0:
        return np.nan
    return np.mean(average_box_directions)


def compute_phase_embedding(x, n_dimensions, delay):
    """
    Compute the phase embedding of a time series

    Parameters
    ----------
    x : np.array
        The time series
    n_dimensions : int
        The number of dimensions of the embedding
    delay : int
        The delay

    Returns
    -------
    np.array
        The phase embedding, shape (N - (n_dimensions - 1) * delay, n_dimensions)
    """
    n_points = len(x)

    n_embedding = n_points - (n_dimensions - 1) * delay
    embedding = np.zeros((n_embedding, n_dimensions))
    for i in range(n_dimensions):
        embedding[:, i] = x[i * delay : i * delay + n_embedding]

    return embedding


def compute_optimal_delay(x, n_bins=30, max_tau=500, ax=None):
    """
    Compute the optimal delay for embedding using entropy.
    arameters
    ----------
    x : np.array
        The time series
    n_bins : int
        The number of bins for computing the entropy
    max_tau : int
        The maximum value of tau
    ax : plt.Axes | None, optional
        The axes on which to plot, by default None

    Returns
    -------
    int
        The optimal embedding delay
    """
    min_x = np.min(x)
    max_x = np.max(x)

    bins = np.linspace(min_x, max_x, n_bins)
    bin_indices = np.digitize(x, bins) - 1

    bin_indices[bin_indices == n_bins - 1] = n_bins - 2
    probabilities = bin_indices / len(x)

    informations = np.zeros(max_tau)
    for tau in range(max_tau):
        # compute the conditional probabilities
        p_tau_hk = np.zeros((n_bins, n_bins))
        for i in range(0, len(x) - tau):
            p_tau_hk[bin_indices[i], bin_indices[i + tau]] += 1
        p_tau_hk /= len(x) - tau

        for h in range(n_bins):
            p_h = probabilities[h]
            for k in range(n_bins):
                p_k = probabilities[k]
                if p_tau_hk[h, k] > 0 and p_h > 0 and p_k > 0:
                    informations[tau] += p_tau_hk[h, k] * np.log(
                        p_tau_hk[h, k] / (p_h * p_k)
                    )

    # find the first minimum
    diff_informations = np.diff(informations)

    if np.all(diff_informations > 0):
        return None
    idx_min = np.where(diff_informations > 0)[0][0]

    if ax is not None:
        ax.plot(informations)
        # vertical line
        ax.axvline(idx_min, color="red", linestyle="--")
        ax.set_xlabel("$\\tau$")
        ax.set_ylabel("$I$")
        ax.grid(color="gray", linestyle="--", linewidth=0.5)

    return idx_min


def find_nearest_neighbors(embedding, epsilon, eps_rate=1.41, perc_waste=10):
    """
    Find the nearest neighbors of each point in the embedding

    Parameters
    ----------
    embedding : np.array
        The embedding, shape (n_points, n_dimensions)
    epsilon : float
        The epsilon
    eps_rate : float
        The rate of increase of epsilon
    perc_waste : float
        The percentage of waste

    Returns
    -------
    np.array
        The index of the nearest neighbors
    np.array
        The distance to the nearest neighbors
    int
        The number of nearest neighbors
    """

    n_points = len(embedding)

    # Compute the distance matrix
    distances = np.full((n_points, n_points), np.inf)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(embedding[i] - embedding[j])
            distances[i, j] = dist
            distances[j, i] = dist

    while True:
        # Find the nearest neighbors
        index_nearest = np.full(n_points, -1)
        distance_nearest = np.full(n_points, np.inf)
        for i in range(n_points):
            for j in range(n_points):
                if (
                    distances[i, j] < epsilon
                    and i != j
                    and distances[i, j] < distance_nearest[i]
                ):
                    distance_nearest[i] = distances[i, j]
                    index_nearest[i] = j

        # number of nearest neighbors
        n_nearest = np.sum(index_nearest != -1)
        percentage_without_nearest = 100 * (1 - n_nearest / n_points)

        if percentage_without_nearest < perc_waste:
            break
        else:
            epsilon *= eps_rate

    return index_nearest, distance_nearest, n_nearest


def compute_optimal_embedding_dimension(
    x, max_dim=20, delay=70, epsilon=0.07, threshold=0.01, ax=None
):
    """
    Compute the optimal embedding dimension using the false nearest neighbors method

    Parameters
    ----------
    x : np.array
        The time series
    max_dim : int
        The maximum embedding dimension
    delay : int
        The delay
    epsilon : float
        The epsilon
    threshold : float
        The threshold for the false nearest neighbors

    Returns
    -------
    int
        The optimal embedding dimension
    """

    embedding = compute_phase_embedding(x, 1, delay)
    prev_nearest_neighbors, prev_distances_nearest, n_nearest = find_nearest_neighbors(
        embedding,
        epsilon,
    )

    max_dim_allowed = int((len(x) - 1) / delay + 1)

    fnn = []
    for m in range(2, min(max_dim, max_dim_allowed) + 1):
        embedding = compute_phase_embedding(x, m, delay)

        n_false_nearest = 0
        for i in range(len(embedding)):
            if prev_nearest_neighbors[i] != -1 and prev_nearest_neighbors[i] < len(
                embedding
            ):
                Ri = np.abs(
                    (embedding[i, -1] - embedding[prev_nearest_neighbors[i], -1])
                    / prev_distances_nearest[i]
                )
                if Ri > 10:
                    n_false_nearest += 1
        fnn.append(n_false_nearest / n_nearest)

        prev_nearest_neighbors, prev_distances_nearest, n_nearest = (
            find_nearest_neighbors(embedding, epsilon)
        )

    # find the value of m where the FNN is below the threshold
    idx_min = np.where(np.array(fnn) < threshold)[0][0]

    if ax is not None:
        ax.plot(fnn)
        # horizontal line
        ax.axhline(threshold, color="red", linestyle="--")
        ax.set_xlabel("$m$")
        ax.set_ylabel("$f_{false}$")
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        # only int values
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return idx_min
