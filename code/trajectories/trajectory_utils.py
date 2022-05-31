import numpy as np
from regex import R


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


def filter_pedestrian(pedestrian, threshold):
    value = threshold.get_value()
    min_val = threshold.get_min_value()
    max_val = threshold.get_max_value()
    aggregate = threshold.is_aggregate()

    column = pedestrian.get_trajectory_column(value)

    if not aggregate:
        if min_val and max_val:
            threshold_indices = np.where((column >= min_val) & (column <= max_val))[0]
        elif min_val:
            threshold_indices = np.where(column >= min_val)[0]
        else:
            threshold_indices = np.where(column <= max_val)[0]

        if len(threshold_indices) > 0:
            trajectory = pedestrian.get_trajectory()[threshold_indices, :]
            pedestrian.set_trajectory(trajectory)
            return pedestrian
        else:
            return None

    else:
        aggregate_val = column[-1] - column[0]
        if (
            (
                min_val
                and max_val
                and aggregate_val >= min_val
                and aggregate_val <= max_val
            )
            or (min_val and aggregate_val >= min_val)
            or (max_val and aggregate_val <= max_val)
        ):
            return pedestrian
        else:
            return None


def filter_pedestrians(pedestrians, threshold):

    filtered_pedestrians = []
    for pedestrian in pedestrians:
        pedestrian = filter_pedestrian(pedestrian, threshold)
        if pedestrian:
            filtered_pedestrians += [pedestrian]

    return filtered_pedestrians
