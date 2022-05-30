import os

from sympy import Add

from utils import *
from constants import *


if __name__ == "__main__":

    dir_path = "../../data/unformatted/atc/annotations/"

    for day in DAYS_ATC:

        group_annotations = os.path.join(dir_path, f"tani_tutti_{day}.pkl")
        group_data = pickle_load(group_annotations)

        social_relation_annotations = os.path.join(dir_path, f"st1_{day}.pkl")
        soc_rel_data = pickle_load(social_relation_annotations)

        groups_data = {}
        individuals_data = {}

        # parse tani_tutti for additional group data (wheelchair, baby carrier)
        additional_data = {}
        for data_row in group_data:
            ped_id = data_row[0]
            group_size = data_row[2]

            additional_data[ped_id] = {
                "wheelchair": data_row[4],
                "electric_wheelchair": data_row[5],
                "stroller": data_row[6],
            }

        for data_row in soc_rel_data:

            ped_id = data_row[0]
            group_size = data_row[1]

            group_members = data_row[2 : 2 + group_size - 1]
            soc_rel = data_row[2 + group_size - 1]
            age = data_row[2 + group_size]
            gender = data_row[2 + group_size + 1]
            purpose = data_row[2 + group_size + 2]

            all_group_members = sorted(group_members + [ped_id])

            group_id = int("".join([str(pid) for pid in all_group_members]))

            if group_id not in groups_data:
                groups_data[group_id] = {
                    "size": group_size,
                    "soc_rel": soc_rel,
                    "purpose": purpose,
                    "members": all_group_members,
                    "additional_data": additional_data[ped_id],
                }

            else:
                if additional_data[ped_id] != groups_data[group_id]["additional_data"]:
                    # print(
                    #     f"Additional data does not match for {group_id}. {ped_id} annotated with {additional_data[ped_id]} while previous member was annotated with {groups_data[group_id]['additional_data']}."
                    # )
                    # keeping the one with the most annotation, would be better to check with the videos
                    new_additional_data_sum = sum(additional_data[ped_id].values())
                    old_additional_data_sum = sum(
                        groups_data[group_id]["additional_data"].values()
                    )
                    if new_additional_data_sum > old_additional_data_sum:
                        groups_data[group_id]["additional_data"] = additional_data[
                            ped_id
                        ]

            if ped_id not in individuals_data:
                individuals_data[ped_id] = {
                    "age": age,
                    "gender": gender,
                    "purpose": purpose,
                    "groups": [group_id],
                }
            # else:
            #     print(
            #         f"Conflicting information for {ped_id}, skipping new data. It would be better to check the videos"
            #     )

        groups_annotations_path = (
            f"../../data/formatted/atc/groups_annotations_{day}.pkl"
        )
        pickle_save(groups_annotations_path, groups_data)

        individuals_annotations_path = (
            f"../../data/formatted/atc/individuals_annotations_{day}.pkl"
        )
        pickle_save(individuals_annotations_path, individuals_data)
        print(
            f"Saving {len(groups_data)} groups annotations to {groups_annotations_path} and {len(individuals_data)} individuals annotations to {individuals_annotations_path}."
        )
