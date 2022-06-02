from trajectories.environment import Environment
from trajectories.plot_utils import *

if __name__ == "__main__":

    atc = Environment("atc", data_dir="../data/formatted")

    all_pedestrians = atc.get_pedestrians(days=["0109"], no_groups=True)

    for ped in all_pedestrians:
        encounters = ped.get_encounters(1000, all_pedestrians)

        if encounters:
            plot_animated_2D_trajectories([ped] + encounters, simultaneous=False)
