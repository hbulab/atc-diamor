from package.src.pedestrians_social_binding.environment import Environment
from package.src.pedestrians_social_binding.constants import *
from package.src.pedestrians_social_binding.trajectory_utils import *
from package.src.pedestrians_social_binding.plot_utils import *
from package.src.pedestrians_social_binding.threshold import Threshold

import matplotlib.pyplot as plt

import scienceplots

plt.style.use("science")


if __name__ == "__main__":

    threshold_v = Threshold("v", min=500, max=3000)  # velocity in [0.5; 3]m/s

    atc = Environment("atc", data_dir="../data/formatted")
    groups_atc = atc.get_groups_grouped_by(
        "soc_rel", size=2, ped_thresholds=[threshold_v]
    )

    diamor = Environment("diamor", data_dir="../data/formatted", raw=True)
    groups_diamor = diamor.get_groups_grouped_by(
        "interaction", size=2, ped_thresholds=[threshold_v]
    )

    # ========================================================
    # Plotting the PDF of the interpersonal distances
    # ========================================================

    N_BINS = 40
    D_MIN = 0.4
    D_MAX = 2
    bin_size = (D_MAX - D_MIN) / N_BINS
    pdf_edges = np.linspace(D_MIN, D_MAX, N_BINS + 1)
    bin_centers = 0.5 * (pdf_edges[0:-1] + pdf_edges[1:])

    # # ------ For ATC ------

    # distances_wrt_soc_rel = {}

    # fig, ax = plt.subplots(figsize=(6, 4))

    # for soc_rel in range(1, len(SOCIAL_RELATIONS_EN)):
    #     n_groups = len(groups_atc[soc_rel])

    #     if soc_rel not in distances_wrt_soc_rel:
    #         distances_wrt_soc_rel[soc_rel] = []

    #     for group in groups_atc[soc_rel]:

    #         distances = group.get_interpersonal_distance()

    #         distances_wrt_soc_rel[soc_rel] = np.concatenate(
    #             (distances_wrt_soc_rel[soc_rel], distances / 1000)
    #         )

    #     # print(len(distances_wrt_soc_rel[soc_rel]))
    #     hist = np.histogram(distances_wrt_soc_rel[soc_rel], pdf_edges)[0]
    #     pdf = hist / sum(hist) / bin_size

    #     ax.plot(bin_centers, pdf, label=f"{SOCIAL_RELATIONS_EN[soc_rel]}", linewidth=2)

    # ax.legend()
    # ax.set_xlabel("$d$ [m]")
    # ax.set_ylabel("p($d$)")
    # ax.legend()
    # ax.grid(color="gray", linestyle="--", linewidth=0.5)

    # plt.show()
    # # plt.savefig("atc_interpersonal_distances.pdf")
    # # plt.close()

    # # ------ For DIAMOR ------

    # distances_wrt_interaction = {}

    # fig, ax = plt.subplots(figsize=(6, 4))

    # for interaction in range(4):
    #     n_groups = len(groups_diamor[interaction])

    #     if interaction not in distances_wrt_interaction:
    #         distances_wrt_interaction[interaction] = []

    #     for group in groups_diamor[interaction]:

    #         distances = group.get_interpersonal_distance()

    #         # plot_static_2D_trajectories(
    #         #     members, boundaries=diamor.boundaries, simultaneous=True
    #         # )

    #         distances_wrt_interaction[interaction] = np.concatenate(
    #             (distances_wrt_interaction[interaction], distances / 1000)
    #         )

    #     # print(len(distances_wrt_interaction[interaction]))
    #     hist = np.histogram(distances_wrt_interaction[interaction], pdf_edges)[0]
    #     pdf = hist / sum(hist) / bin_size

    #     ax.plot(bin_centers, pdf, label=interaction, linewidth=2)

    # ax.set_xlabel("$d$ [m]")
    # ax.set_ylabel("p($d$)")
    # ax.legend()
    # ax.grid(color="gray", linestyle="--", linewidth=0.5)

    # # plt.show()
    # plt.savefig("diamor_interpersonal_distances.pdf")
    # plt.close()

    # ========================================================
    # Plotting the PDF of the velocities
    # ========================================================
    N_BINS = 40
    V_MIN = 0
    V_MAX = 2.5
    bin_size = (V_MAX - V_MIN) / N_BINS
    pdf_edges = np.linspace(V_MIN, V_MAX, N_BINS + 1)

    # # ------ For ATC ------

    # velocities_wrt_soc_rel = {}

    # fig, ax = plt.subplots(figsize=(6, 4))

    # for soc_rel in range(1, len(SOCIAL_RELATIONS_EN)):
    #     n_groups = len(groups_atc[soc_rel])

    #     if soc_rel not in velocities_wrt_soc_rel:
    #         velocities_wrt_soc_rel[soc_rel] = []

    #     for group in groups_atc[soc_rel]:

    #         trajectory = group.get_as_individual().trajectory
    #         if len(trajectory) == 0:
    #             continue

    #         velocities = np.linalg.norm(compute_velocity(trajectory), axis=1)

    #         velocities_wrt_soc_rel[soc_rel] = np.concatenate(
    #             (velocities_wrt_soc_rel[soc_rel], velocities)
    #         )

    #     # print(len(velocities_wrt_soc_rel[soc_rel]))
    #     hist = np.histogram(velocities_wrt_soc_rel[soc_rel], pdf_edges)[0]
    #     pdf = hist / sum(hist) / bin_size

    #     ax.plot(
    #         pdf_edges[0:-1], pdf, label=f"{SOCIAL_RELATIONS_EN[soc_rel]}", linewidth=2
    #     )

    # ax.legend()
    # ax.set_xlabel("$v$ [m/s]")
    # ax.set_ylabel("p($v$)")
    # ax.legend()
    # ax.grid(color="gray", linestyle="--", linewidth=0.5)

    # # plt.show()
    # plt.savefig("atc_velocities.pdf")
    # plt.close()

    # ------ For DIAMOR ------

    velocities_wrt_interaction = {}

    fig, ax = plt.subplots(figsize=(6, 4))

    for interaction in range(4):
        n_groups = len(groups_diamor[interaction])

        if interaction not in velocities_wrt_interaction:
            velocities_wrt_interaction[interaction] = []

        for group in groups_diamor[interaction]:

            trajectory = group.get_as_individual().trajectory

            if len(trajectory) == 0:
                continue

            velocities = np.linalg.norm(compute_velocity(trajectory), axis=1) / 1000
            velocities_wrt_interaction[interaction] = np.concatenate(
                (velocities_wrt_interaction[interaction], velocities)
            )

        # print(len(velocities_wrt_interaction[interaction]))
        hist = np.histogram(velocities_wrt_interaction[interaction], pdf_edges)[0]
        pdf = hist / sum(hist) / bin_size

        ax.plot(pdf_edges[0:-1], pdf, label=interaction, linewidth=2)

    ax.set_xlabel("$v$ [m/s]")
    ax.set_ylabel("p($v$)")
    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.show()
    # plt.savefig("diamor_velocities.pdf")
    # plt.close()
