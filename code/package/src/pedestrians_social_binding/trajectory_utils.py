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
            condition = d_AB >= min_val and d_AB <= max_val
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
        np.cross(end_point - start_point, middle_points - start_point)
    ) / np.linalg.norm(end_point - start_point)
    if scaled:  # divide by the distance from P to E
        distances_to_straight_line /= np.linalg.norm(end_point - start_point)

    max_distance = np.max(distances_to_straight_line)

    return max_distance


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
    start_point = position[0]
    end_point = position[-1]
    net_dislacement = np.linalg.norm(end_point - start_point)
    gross_displacement = np.sum(np.linalg.norm(position[:-1] - position[1:], axis=1))
    # print(trajectory_length)
    return net_dislacement / gross_displacement


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
        "maximum_scaled_lateral_deviation", "maximum_lateral_deviation", "sinuosity"

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
