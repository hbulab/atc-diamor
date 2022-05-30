import numpy as np
import os

from utils import *

if __name__ == "__main__":

    data_path = "../../data/raw/diamor"

    for data_index in [1, 2]:
        dir_name = f"DIAMOR-{data_index}"
        dir_path = os.path.join(data_path, dir_name)

        groups_file_name = f"groups_DIAMOR-{data_index}.dat"
        traj_file_name = f"person_DIAMOR-{data_index}_all.csv"

        groups_file_path = os.path.join(dir_path, groups_file_name)

        # process the groups info

        # with open(groups_file_path, "r") as f:
        #     for line in f.readlines():
        #         data = list(map(int, line.rstrip("\n").rstrip(" ").split()))
        #         track_type, ped_id, group_members_ids, interacting_i = parse_group_data(
        #             data
        #         )

        trajectories = {}
        traj_file_path = os.path.join(dir_path, traj_file_name)
        with open(traj_file_path, "r") as f:
            for line in f.readlines():
                data = list(map(float, line.rstrip("\n").rstrip(" ").split(",")))
                time_stamp = data[0]
                ped_id = int(data[1])
                pos_x, pos_y, pos_z = int(data[2]), int(data[3]), int(data[4])
                vel = data[5]
                angle_of_motion = data[6]
                facing_angle = data[7]

                if ped_id not in trajectories:
                    trajectories[ped_id] = []

                trajectories[ped_id] += [
                    time_stamp,
                    pos_x,
                    pos_y,
                    pos_z,
                    vel,
                    angle_of_motion,
                    facing_angle,
                ]
