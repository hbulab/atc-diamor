import numpy as np
import pandas as pd
from pathlib import Path
from utils import pickle_save


def get_annotations_dict(annotations):
    # col 1: pedestrian id
    # col 2: size of the group
    # col 3-8: ids of the other group members
    # col 9: number of pedestrians interacting with this pedestrians
    # col 10-14: ids of the other pedestrians interacting with this pedestrian
    # col 15-18: type of interaction

    annotations_dict = {}
    sizes = {}
    for row in annotations:
        ped_id = int(row[0])
        if ped_id in annotations_dict:
            raise ValueError(f"Ped {ped_id} already in dict")
        group_size = int(row[1])
        if group_size not in sizes:
            sizes[group_size] = 0
        sizes[group_size] += 1
        annotations_dict[ped_id] = {
            "group_size": int(row[1]),
            "group_members": [int(row[i]) for i in range(2, 8) if int(row[i])],
            "num_interacting": int(row[8]),
            "interacting_ped_ids": [int(row[i]) for i in range(9, 14) if int(row[i])],
            "interaction_type": row[14:],
            "is_interacting": np.any(row[14:]),
        }
    print(f"Group sizes: {sizes}")
    return annotations_dict


if __name__ == "__main__":
    data_path = Path("../../data/unformatted/diamor/annotations")
    granular_annotations_tani_path = data_path / "taniguchi_gt_gest_06.csv"
    granular_annotations_tani = pd.read_csv(granular_annotations_tani_path).to_numpy()

    annotations_dict_tani = get_annotations_dict(granular_annotations_tani)

    # save annotations
    path_to_save = data_path / "taniguchi_gt_gest_06.pkl"
    pickle_save(path_to_save, annotations_dict_tani)
