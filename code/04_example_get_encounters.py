from trajectories.environment import Environment
from trajectories.threshold import Threshold
from trajectories.plot_utils import *

if __name__ == "__main__":

    atc = Environment("atc", data_dir="../data/formatted")

    threshold_v = Threshold("v", 500, 3000)

    all_pedestrians = atc.get_pedestrians(
        days=["0109"], no_groups=True, thresholds=[threshold_v]
    )

    v = 4000

    for ped in all_pedestrians:
        encounters = ped.get_encounters(v, all_pedestrians)

        if encounters:
            plot_animated_2D_trajectories(
                [ped] + encounters,
                boundaries=atc.boundaries,
                vicinity=v,
                simultaneous=False,
            )
