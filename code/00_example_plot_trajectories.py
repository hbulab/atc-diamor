from trajectories.environment import Environment

if __name__ == "__main__":

    # -------- ATC --------
    atc = Environment("atc", data_dir="../data/formatted")

    all_pedestrians = atc.get_pedestrians()
    n_pedestrians = len(all_pedestrians)

    non_groups = atc.get_pedestrians(no_groups=True)
    n_non_groups = len(non_groups)

    print(
        f"Found {n_non_groups} non-groups out of {n_pedestrians} pedestrians ({n_pedestrians - n_non_groups} group members)."
    )

    all_groups = atc.get_groups_grouped_by("size")
    tot_group_member = 0
    for size, groups in all_groups.items():
        print(f"{len(groups)} groups of size {size}.")
        tot_group_member += size * len(groups)
    print(f" -> Found {tot_group_member} groups members.")

    for pedestrian in non_groups[:3]:
        pedestrian.plot_2D_trajectory(animate=False, scale=False)

    # -------- DIAMOR --------
    diamor = Environment("diamor", data_dir="../data/formatted")
    pedestrians = diamor.get_pedestrians(days=["06"])

    for pedestrian in pedestrians[:3]:

        pedestrian.plot_2D_trajectory(animate=True, scale=True)
