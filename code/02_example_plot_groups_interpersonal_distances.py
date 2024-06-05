from package.src.pedestrians_social_binding.environment import Environment
from package.src.pedestrians_social_binding.constants import *
from package.src.pedestrians_social_binding.trajectory_utils import *
from package.src.pedestrians_social_binding.plot_utils import *

import matplotlib.pyplot as plt

import scienceplots

plt.style.use("science")


if __name__ == "__main__":

    N_BINS = 40
    D_MIN = 0.4
    D_MAX = 2
    bin_size = (D_MAX - D_MIN) / N_BINS
    pdf_edges = np.linspace(D_MIN, D_MAX, N_BINS + 1)
    bin_centers = 0.5 * (pdf_edges[0:-1] + pdf_edges[1:])

    # ------ For ATC ------

    atc = Environment("atc", data_dir="../data/formatted")
    groups_atc = atc.get_groups_grouped_by("soc_rel", size=2)

    distances_wrt_soc_rel = {}

    for soc_rel in range(1, len(SOCIAL_RELATIONS_EN)):
        n_groups = len(groups_atc[soc_rel])

        if soc_rel not in distances_wrt_soc_rel:
            distances_wrt_soc_rel[soc_rel] = []

        for group in groups_atc[soc_rel]:

            distances = group.get_interpersonal_distance()

            distances_wrt_soc_rel[soc_rel] = np.concatenate(
                (distances_wrt_soc_rel[soc_rel], distances / 1000)
            )

        # print(len(distances_wrt_soc_rel[soc_rel]))
        hist = np.histogram(distances_wrt_soc_rel[soc_rel], pdf_edges)[0]
        pdf = hist / sum(hist) / bin_size

        plt.plot(bin_centers, pdf, label=f"{SOCIAL_RELATIONS_EN[soc_rel]}")

    plt.legend()
    plt.xlabel("d (m)")
    plt.ylabel("p(d)")
    plt.legend()

    plt.show()

    # ------ For DIAMOR ------

    diamor = Environment("diamor", data_dir="../data/formatted")
    groups_diamor = diamor.get_groups_grouped_by("interaction", size=2)

    distances_wrt_interaction = {}

    with plt.rc_context({"font.size": 14}):

        fig, ax = plt.subplots(figsize=(6, 4))

        for interaction in range(4):
            n_groups = len(groups_diamor[interaction])

            if interaction not in distances_wrt_interaction:
                distances_wrt_interaction[interaction] = []

            for group in groups_diamor[interaction]:

                distances = group.get_interpersonal_distance()

                # plot_static_2D_trajectories(
                #     members, boundaries=diamor.boundaries, simultaneous=True
                # )

                distances_wrt_interaction[interaction] = np.concatenate(
                    (distances_wrt_interaction[interaction], distances / 1000)
                )

            # print(len(distances_wrt_interaction[interaction]))
            hist = np.histogram(distances_wrt_interaction[interaction], pdf_edges)[0]
            pdf = hist / sum(hist) / bin_size

            ax.plot(bin_centers, pdf, label=interaction)

        ax.set_xlabel("$d$ (m)")
        ax.set_ylabel("p($d$)")
        ax.legend()
        ax.grid(color="gray", linestyle="--", linewidth=0.5)

        plt.show()
