import os

import pandas as pd

from utils import *
from constants import *

if __name__ == "__main__":
    dir_path = "../../data/unformatted/diamor/annotations/"

    for day in DAYS_DIAMOR:
        group_annotations = os.path.join(
            dir_path, f"ids_wrt_group_size_taniguchi_{day}.pkl"
        )
        group_data = pickle_load(group_annotations)

        groups_data = {}
        individuals_data = {}

        for groups in group_data:
            for group in groups:
                group_members = sorted(list(map(int, group)))
                group_id = int("".join([str(int(pid)) for pid in group_members]))

                if group_id not in groups_data:
                    groups_data[group_id] = {
                        "size": len(group),
                        "members": group_members,
                    }

                for ped_id in group_members:
                    if ped_id not in individuals_data:
                        individuals_data[ped_id] = {
                            "groups": [],
                        }
                    individuals_data[ped_id]["groups"] += [group_id]

        interaction_annotations = os.path.join(dir_path, f"gt_2p_yoshioka_{day}.pkl")
        interaction_data = pickle_load(interaction_annotations)

        for row in interaction_data:
            group_members = sorted([int(row[0]), int(row[1])])
            group_id = int("".join([str(int(pid)) for pid in group_members]))
            if group_id not in groups_data:
                groups_data[group_id] = {
                    "size": 2,
                    "members": group_members,
                    "interaction": row[4],
                }
            else:
                groups_data[group_id]["interaction"] = row[4]

            for ped_id in group_members:
                if ped_id not in individuals_data:
                    individuals_data[ped_id] = {
                        "groups": [],
                    }
                individuals_data[ped_id]["groups"] += [group_id]

        # load granular annotations
        granular_annotations_path = os.path.join(dir_path, f"taniguchi_gt_gest_06.pkl")
        granular_annotations = pickle_load(granular_annotations_path)
        for group_id in groups_data:
            members = groups_data[group_id]["members"]
            interactions = {}
            for member in members:
                if member in granular_annotations:
                    # compare size of groups
                    if (
                        groups_data[group_id]["size"]
                        != granular_annotations[member]["group_size"]
                    ):
                        continue  # skipping group because of size mismatch
                    interactions[member] = {
                        "is_interacting": granular_annotations[member][
                            "is_interacting"
                        ],
                        "interaction_type": granular_annotations[member][
                            "interaction_type"
                        ],
                    }
            groups_data[group_id]["interactions"] = interactions

        groups_annotations_path = (
            f"../../data/formatted/diamor/groups_annotations_{day}.pkl"
        )
        pickle_save(groups_annotations_path, groups_data)
        individuals_annotations_path = (
            f"../../data/formatted/diamor/individuals_annotations_{day}.pkl"
        )
        pickle_save(individuals_annotations_path, individuals_data)
        print(
            f"Saving {len(groups_data)} groups annotations to {groups_annotations_path} and {len(individuals_data)} individuals annotations to {individuals_annotations_path}."
        )
