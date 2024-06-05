import os

import pandas as pd

from utils import *
from constants import *

if __name__ == "__main__":
    dir_path = "../../data/unformatted/diamor/annotations/"

    for day in DAYS_DIAMOR:
        print(f"Day: {day}")
        group_annotations = os.path.join(
            dir_path, f"ids_wrt_group_size_taniguchi_{day}.pkl"
        )
        raw_group_data = pickle_load(group_annotations)

        groups_data = {}
        individuals_data = {}

        for groups in raw_group_data:  # loop over group sizes
            # if len(groups) > 0:
            #     print(f"{len(groups[0])} -> {len(groups)}")

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

        n_soc = {}
        overlap = 0
        for row in interaction_data:
            group_members = sorted([int(row[0]), int(row[1])])
            group_id = int("".join([str(int(pid)) for pid in group_members]))
            interaction = row[4]
            if interaction not in n_soc:
                n_soc[interaction] = 0
            n_soc[interaction] += 1

            if group_id not in groups_data:
                groups_data[group_id] = {
                    "size": 2,
                    "members": group_members,
                    "interaction": interaction,
                }
            else:
                overlap += 1
                groups_data[group_id]["interaction"] = interaction

            for ped_id in group_members:
                if ped_id not in individuals_data:
                    individuals_data[ped_id] = {
                        "groups": [],
                    }
                individuals_data[ped_id]["groups"] += [group_id]
        # print(overlap)
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
