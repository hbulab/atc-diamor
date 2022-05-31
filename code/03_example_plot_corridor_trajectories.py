from trajectories.environment import Environment
from trajectories.threshold import Threshold

if __name__ == "__main__":

    atc = Environment("atc", data_dir="../data/formatted")

    # add thresholds to keep only corridor data
    threshold_x = Threshold("x", 5000, 48000)
    threshold_y = Threshold("y", -27000, 8000)

    pedestrians = atc.get_pedestrians(
        days=["0508"], thresholds=[threshold_x, threshold_y]
    )

    for pedestrian in pedestrians[:3]:
        pedestrian.plot_2D_trajectory(animate=False, scale=False)
