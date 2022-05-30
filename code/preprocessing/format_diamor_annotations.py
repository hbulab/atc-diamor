import os

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

        for groups in group_data:
            for group in groups:
                group_members = sorted(group)
                group_id = int("".join([str(int(pid)) for pid in group_members]))

                if group_id not in groups_data:
                    groups_data[group_id] = {
                        "size": len(group),
                    }

        interaction_annotations = os.path.join(dir_path, f"gt_2p_yoshioka_{day}.pkl")
        interaction_data = pickle_load(interaction_annotations)

        for row in interaction_data:
            group_members = sorted([row[0], row[1]])
            group_id = int("".join([str(int(pid)) for pid in group_members]))
            if group_id not in groups_data:
                groups_data[group_id] = {"size": 2, "interaction": row[4]}
            else:
                groups_data[group_id]["interaction"] = row[4]

        groups_annotations_path = (
            f"../../data/formatted/diamor/groups_annotations_{day}.pkl"
        )
        pickle_save(groups_annotations_path, groups_data)
        print(
            f"Saving {len(groups_data)} groups annotations to {groups_annotations_path}."
        )
