from package.src.pedestrians_social_binding.environment import Environment

if __name__ == "__main__":

    # -------- DIAMOR --------
    diamor = Environment("diamor", data_dir="../data/formatted", raw=True)
    pedestrians = diamor.get_pedestrians(days=["06"])

    for pedestrian in pedestrians[:3]:
        pedestrian.plot_2D_trajectory(animate=True, scale=True)
