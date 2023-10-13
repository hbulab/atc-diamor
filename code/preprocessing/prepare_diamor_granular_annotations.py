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
    for row in annotations:
        ped_id = int(row[0])
        if ped_id in annotations_dict:
            raise ValueError(f"Ped {ped_id} already in dict")
        annotations_dict[ped_id] = {
            "group_size": int(row[1]),
            "group_members": [int(row[i]) for i in range(2, 8) if int(row[i])],
            "num_interacting": int(row[8]),
            "interacting_ped_ids": [int(row[i]) for i in range(9, 14) if int(row[i])],
            "interaction_type": row[14:],
            "is_interacting": np.any(row[14:]),
        }
    return annotations_dict


if __name__ == "__main__":
    data_path = Path("../../data/unformatted/diamor/annotations")
    granular_annotations_tani_path = data_path / "taniguchi_gt_gest_06.csv"
    granular_annotations_tani = pd.read_csv(granular_annotations_tani_path).to_numpy()
    granular_annotations_tomita_path = data_path / "tomita_gt_gest_06.csv"
    granular_annotations_tomita = pd.read_csv(
        granular_annotations_tomita_path
    ).to_numpy()

    annotations_dict_tani = get_annotations_dict(granular_annotations_tani)
    annotations_dict_tomita = get_annotations_dict(granular_annotations_tomita)

    # check agreement
    for ped_id in annotations_dict_tani:
        if ped_id not in annotations_dict_tomita:
            print(f"Ped {ped_id} not in Tomita annotations")
            continue
        if (
            annotations_dict_tani[ped_id]["is_interacting"]
            != annotations_dict_tomita[ped_id]["is_interacting"]
        ):
            print(
                f"Ped {ped_id} has different is_interacting values in Tomita and Tani"
            )
        # check

    # save annotations
    path_to_save = data_path / "taniguchi_gt_gest_06.pkl"
    pickle_save(path_to_save, annotations_dict_tani)
