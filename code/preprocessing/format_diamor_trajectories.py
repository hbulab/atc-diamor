import numpy as np
import os

from utils import *
from constants import *

if __name__ == "__main__":

    dir_path = "../../data/unformatted/diamor/trajectories/"

    for day in DAYS_DIAMOR:

        daily_traj = {}

        position_path = os.path.join(dir_path, f"crowd_{day}.pkl")
        velocity_path = os.path.join(dir_path, f"crowd_velocity_{day}.pkl")

        position_data = pickle_load(position_path)
        velocity_data = pickle_load(velocity_path)

        discarded = 0

        for ped_id in map(int, position_data.keys()):

            times = position_data[str(ped_id)]["time"]
            pos_xy = position_data[str(ped_id)]["traj"]
            pos_z = np.full((times.shape), 0.0)
            vel = velocity_data[str(ped_id)]["velocity"]
            vel_x = velocity_data[str(ped_id)]["vx"]
            vel_y = velocity_data[str(ped_id)]["vy"]

            if len(times) > 3:  # need at least four data points

                # if pos contains N elements, vel contains N - 2

                traj = np.concatenate(
                    (times[:-2], pos_xy[:-2], pos_z[:-2], vel, vel_x, vel_y), axis=1
                )
                print(traj.shape)

                daily_traj[ped_id] = traj

            else:
                # print(f"{ped_id} doesn't have enough data points (less than 4).")
                discarded += 1
        formatted_traj_path = f"../../data/formatted/diamor/trajectories_{day}.pkl"
        pickle_save(formatted_traj_path, daily_traj)
        print(
            f"Saving {len(daily_traj)} trajectories to {formatted_traj_path}. {discarded} discarded trajectories."
        )
