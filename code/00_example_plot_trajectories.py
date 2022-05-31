from trajectories.environment import Environment

if __name__ == "__main__":

    atc = Environment("atc", data_dir="../data/formatted")
    pedestrians = atc.get_pedestrians(days=["0508"])

    for pedestrian in pedestrians[:3]:

        pedestrian.plot_2D_trajectory(animate=True, scale=True)

    diamor = Environment("diamor", data_dir="../data/formatted")
    pedestrians = diamor.get_pedestrians(days=["06"])

    for pedestrian in pedestrians[:3]:

        pedestrian.plot_2D_trajectory(animate=True, scale=True)
