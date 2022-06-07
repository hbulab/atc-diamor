from package.src.pedestrians_social_binding.environment import Environment
from package.src.pedestrians_social_binding.threshold import Threshold
from package.src.pedestrians_social_binding.plot_utils import *

if __name__ == "__main__":

    atc = Environment("atc", data_dir="../data/formatted")

    threshold_v = Threshold("v", 500, 3000)
    threshold_d = Threshold("d", 2000)  # walk at least 2 m
    all_pedestrians = atc.get_pedestrians(
        days=["0109"], no_groups=True, thresholds=[threshold_v, threshold_d]
    )

    v = 4000

    for ped in all_pedestrians:
        encounters = ped.get_encountered_pedestrians(v, all_pedestrians)

        if encounters:
            plot_animated_2D_trajectories(
                [ped] + encounters,
                boundaries=atc.boundaries,
                vicinity=v,
                simultaneous=False,
            )
