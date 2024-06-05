import pandas as pd
from package.src.pedestrians_social_binding.environment import Environment
import matplotlib.pyplot as plt


if __name__ == "__main__":
    diamor = Environment("diamor", data_dir="../data/formatted", raw=True)

    groups = diamor.get_groups(size=2)

    dict_verif_annotations = {i: {0: 0, 1: 0, 2: 0} for i in [0, 1, 2, 3]}
    n = 0
    for group in groups:
        group_size = group.annotations["size"]

        if not "interaction" in group.annotations:
            continue

        group_intensity = group.annotations["interaction"]

        if len(group.annotations["interactions"]) != group.annotations["size"]:
            continue

        n += 1

        n_interacting = sum(
            [
                group.annotations["interactions"][ped_id]["is_interacting"]
                for ped_id in group.annotations["interactions"]
            ]
        )

        dict_verif_annotations[group_intensity][n_interacting] += 1

    print(n)
    df = pd.DataFrame(dict_verif_annotations).T

    df.plot.bar()
    plt.show()
