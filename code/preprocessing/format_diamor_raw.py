from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from tqdm import tqdm
from utils import pickle_save

from constants import DAYS_DIAMOR

if __name__ == "__main__":
    dir_path = Path("../../data/raw/diamor")

    columns_to_keep = [0, 4, 5, 6]

    for day in DAYS_DIAMOR:
        raw_trajectories = {}

        day_dir = dir_path / day

        data_files = list(day_dir.glob("*.dat"))
        for data_file in tqdm(data_files):
            data = np.loadtxt(data_file)
            for row in data:
                ped_id = int(row[2])
                if ped_id not in raw_trajectories:
                    raw_trajectories[ped_id] = []
                raw_trajectories[ped_id] += [row[columns_to_keep]]

        daily_trajectories = {}
        for ped_id in raw_trajectories:
            ped_trajectory = np.array(raw_trajectories[ped_id])
            # sort by time
            ped_trajectory = ped_trajectory[ped_trajectory[:, 0].argsort()]
            pos_xy = ped_trajectory[:, 1:3]
            time = ped_trajectory[:, 0]
            # plt.plot(time)
            # plt.show()
            vel = np.diff(pos_xy, axis=0) / np.diff(time, axis=0)[:, None]
            vel_mag = np.linalg.norm(vel, axis=1)
            full_trajectory = np.concatenate(
                (ped_trajectory[:-1, :], vel_mag[:, None], vel), axis=1
            )
            if len(full_trajectory) > 1:
                daily_trajectories[ped_id] = full_trajectory

        formatted_traj_path = f"../../data/formatted/diamor/trajectories_raw_{day}.pkl"
        pickle_save(formatted_traj_path, daily_trajectories)

        # plt.scatter(
        #     daily_trajectories[ped_id][:, 1], daily_trajectories[ped_id][:, 2]
        # )
        # plt.show()
