from __future__ import annotations
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from pedestrians_social_binding.group import Group
    from pedestrians_social_binding.pedestrian import Pedestrian
    from pedestrians_social_binding.threshold import Threshold

from pedestrians_social_binding.constants import *
from pedestrians_social_binding.parameters import *


import numpy as np
from scipy.spatial import distance


def compute_simultaneous_observations(trajectories):
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


def have_simultaneous_observations(trajectories):
    return len(compute_simultaneous_observations(trajectories)[0]) > 0


def get_trajectory_at_times(trajectory, times):
    times_traj = trajectory[:, 0]
    at_times = np.isin(times_traj, times)
    return trajectory[at_times]


def get_trajectories_at_times(trajectories, times):
    trajectories_at_time = []
    for trajectory in trajectories:
        trajectories_at_time += [get_trajectory_at_times(trajectory, times)]

    return trajectories_at_time


def get_trajectory_not_at_times(trajectory, times):
    times_traj = trajectory[:, 0]
    at_times = np.isin(times_traj, times)
    return trajectory[np.logical_not(at_times)]


def get_trajectories_not_at_times(trajectories, times):
    trajectories_not_at_time = []
    for trajectory in trajectories:
        trajectories_not_at_time += [get_trajectory_not_at_times(trajectory, times)]

    return trajectories_not_at_time


def get_padded_trajectories(trajectories, extend=True):
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


def compute_interpersonal_distance(trajectory_A, trajectory_B):
    """Compute the pair-wise distances between two trajectories

    Parameters
    ----------
    trajectory_A : ndarray
        A trajectory
    trajectory_B : ndarray
        A trajectory

    Returns
    -------
    ndarray
        1D array containing the pair-wise distances
    """

    sim_traj_A, sim_traj_B = compute_simultaneous_observations(
        [trajectory_A, trajectory_B]
    )
    pos_A = sim_traj_A[:, 1:3]
    pos_B = sim_traj_B[:, 1:3]

    d = np.linalg.norm(pos_A - pos_B, axis=1)

    return d


def compute_relative_direction(trajectory_A, trajectory_B):
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

    cos_vA_vB = v_d_dot / (np.linalg.norm(v_A, axis=1) * np.linalg.norm(v_B, axis=1))

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


def filter_pedestrian(pedestrian: Pedestrian, threshold: Threshold):
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

    if value == "t":  # threshold on the tome
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

    column = pedestrian.get_trajectory_column(value)
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
        return None


def filter_pedestrians(pedestrians: list[Pedestrian], threshold: Threshold):

    filtered_pedestrians = []
    for pedestrian in pedestrians:
        pedestrian = filter_pedestrian(pedestrian, threshold)
        if pedestrian:
            filtered_pedestrians += [pedestrian]

    return filtered_pedestrians


def filter_group(group: Group, threshold: Threshold):
    value = threshold.get_value()
    min_val = threshold.get_min_value()
    max_val = threshold.get_max_value()

    if value == "delta":
        d_AB = group.get_interpersonal_distance()

        if min_val is not None and max_val is not None:
            condition = d_AB >= min_val and d_AB <= max_val
        elif min_val is not None:
            condition = d_AB >= min_val
        else:
            condition = d_AB <= max_val
        return np.all(condition)


def translate_position(position, translation):
    return position + translation


def rotate_position(position, angle):
    r_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    r_position = np.dot(r_mat, position.T).T
    return r_position


def compute_interpersonal_distance(pos_A, pos_B):
    dist_AB = np.linalg.norm(pos_A - pos_B, axis=1)
    return dist_AB


def compute_relative_orientation(traj_center_of_mass, traj_A, traj_B):
    v_G = traj_center_of_mass[:, 5:7]
    pos_A = traj_A[:, 1:3]
    pos_B = traj_B[:, 1:3]
    d_AB = pos_B - pos_A
    rel_orientation = np.arctan2(d_AB[:, 1], d_AB[:, 0]) - np.arctan2(
        v_G[:, 1], v_G[:, 0]
    )
    rel_orientation[rel_orientation > np.pi] -= 2 * np.pi
    rel_orientation[rel_orientation < -np.pi] += 2 * np.pi
    return rel_orientation


def compute_continuous_sub_trajectories(trajectory, max_gap=2000):

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


def compute_maximum_lateral_deviation(position, scaled=True):
    start_point = position[0]
    end_point = position[-1]
    middle_points = position[1:-1]
    # for all points except first and last, compute the distance between the line
    # from start S to end E and the point P
    # i.e. (SE x PE) / ||PE||
    distances_to_straight_line = np.abs(
        np.cross(end_point - start_point, middle_points - start_point)
    ) / np.linalg.norm(end_point - start_point)
    if scaled:  # divide by the distance from P to E
        distances_to_straight_line /= np.linalg.norm(end_point - start_point)

    max_distance = np.max(distances_to_straight_line)

    return max_distance


def compute_straightness_index(position):
    start_point = position[0]
    end_point = position[-1]
    net_dislacement = np.linalg.norm(end_point - start_point)
    gross_displacement = np.sum(np.linalg.norm(position[:-1] - position[1:], axis=1))
    # print(trajectory_length)
    return net_dislacement / gross_displacement


def compute_turning_angles(position):
    step_vectors = position[1:, :] - position[:-1, :]
    turning_angles = np.arctan2(step_vectors[1:, 0], step_vectors[1:, 1]) - np.arctan2(
        step_vectors[:-1, 0], step_vectors[:-1, 1]
    )
    turning_angles[turning_angles > np.pi] -= 2 * np.pi
    turning_angles[turning_angles < -np.pi] += 2 * np.pi

    return turning_angles


def rediscretize_position(position):
    step_sizes = np.linalg.norm(position[:-1] - position[1:], axis=1)
    n_points = len(position)
    q = np.min(step_sizes)
    current_goal_index = 1
    current_point = position[0]
    current_goal = position[current_goal_index]
    rediscretized_position = [current_point]
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


def compute_sinuosity(position):
    rediscretized_position = rediscretize_position(position)
    step_size = np.linalg.norm(rediscretized_position[1] - rediscretized_position[0])
    turning_angles = compute_turning_angles(rediscretized_position)
    sinuosity = 1.18 * np.std(turning_angles) / step_size**0.5
    return sinuosity


def compute_deflection(position, measure="straightness_index"):
    if measure == "straightness_index":
        return compute_straightness_index(position)
    elif measure == "maximum_scaled_lateral_deviation":
        return compute_maximum_lateral_deviation(position, scaled=True)
    elif measure == "maximum_lateral_deviation":
        return compute_maximum_lateral_deviation(position, scaled=False)
    elif measure == "sinuosity":
        return compute_sinuosity(position)


def get_pieces(position, piece_size, overlap=False, delta=100):
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


def compute_piecewise_deflections(
    position, piece_size, delta=100, measure="straightness_index"
):
    pieces = get_pieces(position, piece_size, delta=delta)
    deflections = [
        compute_deflection(piece, measure=measure)
        for piece in pieces
        if len(piece) >= 3
    ]
    return deflections
