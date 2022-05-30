import numpy as np
import os

from utils import *
from constants import *

if __name__ == "__main__":

    dir_path = "../../data/unformatted/atc/trajectories/"

    for day in DAYS_ATC:

        daily_traj = {}

        trajectory_path = os.path.join(dir_path, f"smoothed_trajectories_{day}.pkl")
        trajectory_data = pickle_load(trajectory_path)

        i, discarded = 0, 0
        while i < len(trajectory_data):
            trajectory_info = trajectory_data[i]
            ped_id = int(trajectory_info[0])
            n_points = int(trajectory_info[1])
            traj = np.array(
                [np.array(row) for row in trajectory_data[i + 1 : i + 1 + n_points]]
            )

            times = (1000 * traj[:, 0] + traj[:, 1]).reshape(
                (n_points, 1)
            )  # add seconds and milliseconds columns
            pos_xyz = traj[:, 4:7]
            vel = traj[:, 7].reshape((n_points, 1))
            vel_x = traj[:, 8].reshape((n_points, 1))
            vel_y = traj[:, 9].reshape((n_points, 1))

            if len(times) > 3:  # need at least four data points
                # if pos contains N elements, vel contains N - 2

                traj = np.concatenate((times, pos_xyz, vel, vel_x, vel_y), axis=1)
                # print(traj.shape)

                daily_traj[ped_id] = traj

            else:
                # print(f"{ped_id} doesn't have enough data points (less than 4).")
                discarded += 1

            i += n_points + 1

        formatted_traj_path = f"../../data/formatted/atc/trajectories_{day}.pkl"
        pickle_save(formatted_traj_path, daily_traj)
        print(
            f"Saving {len(daily_traj)} trajectories to {formatted_traj_path}. {discarded} discarded trajectories."
        )
