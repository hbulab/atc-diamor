from package.src.pedestrians_social_binding.environment import Environment

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # -------- DIAMOR --------
    diamor = Environment("diamor", data_dir="../data/formatted", raw=True)
    pedestrians = diamor.get_pedestrians(days=["06"])

    all_frequencies = []

    for pedestrian in pedestrians:
        trajectory = pedestrian.get_trajectory()
        times = trajectory[:, 0]
        dt = np.diff(times)
        f = 1 / dt
        all_frequencies += list(f)

    # plot histogram of f
    plt.hist(all_frequencies, bins=100)
    plt.show()
