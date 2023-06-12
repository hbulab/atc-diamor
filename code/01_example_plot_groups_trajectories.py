from package.src.pedestrians_social_binding.environment import Environment

if __name__ == "__main__":

    atc = Environment("atc", data_dir="../data/formatted")
    groups_atc = atc.get_groups(days=["0508"], size=3)

    for group in groups_atc:
        group.as_individual.plot_2D_trajectory(animate=False, scale=True)

    # diamor = Environment("diamor", data_dir="../data/formatted")
    # groups_diamor = diamor.get_groups(days=["06"], size=5)

    # for group in groups_diamor:
    #     print(group.members)
