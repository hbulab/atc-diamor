from lib2to3.pgen2.token import RPAR
import numpy as np


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
