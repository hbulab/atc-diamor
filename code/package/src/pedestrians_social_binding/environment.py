from pedestrians_social_binding.utils import *
from pedestrians_social_binding.trajectory_utils import *
from pedestrians_social_binding.constants import *
from pedestrians_social_binding.pedestrian import Pedestrian
from pedestrians_social_binding.group import Group

import os


class Environment:
    def __init__(self, name, data_dir):

        if name not in ["atc", "diamor"]:
            raise ValueError(f"Unknown environment {name}.")
        self.name = name
        self.data_dir = os.path.join(data_dir, self.name)
        self.boundaries = BOUNDARIES_ATC if self.name == "atc" else BOUNDARIES_DIAMOR
        self.days = DAYS_ATC if self.name == "atc" else DAYS_DIAMOR

    def get_pedestrians(
        self, ids=[], thresholds=[], no_groups=False, days=None
    ) -> list[Pedestrian]:

        if days is None:
            days = self.days

        pedestrians = []
        for day in days:
            traj_path = os.path.join(self.data_dir, f"trajectories_{day}.pkl")
            if not os.path.exists(traj_path):
                raise ValueError(
                    f"Coud not find trajectory data for day {day} in {self.data_dir}."
                )

            daily_traj = pickle_load(traj_path)

            individual_annotations_path = os.path.join(
                self.data_dir, f"individuals_annotations_{day}.pkl"
            )
            if not os.path.exists(individual_annotations_path):
                raise ValueError(
                    f"Coud not find annotation data for day {day} in {self.data_dir}."
                )
            individual_annotations = pickle_load(individual_annotations_path)

            for ped_id in daily_traj:
                if ids and ped_id not in ids:
                    continue
                trajectory = daily_traj[ped_id]
                if ped_id in individual_annotations:
                    groups = individual_annotations[ped_id]["groups"]
                else:
                    groups = []

                if no_groups and len(groups) > 0:
                    continue

                pedestrian = Pedestrian(ped_id, self, day, trajectory, groups)
                pedestrians += [pedestrian]

        # apply the potential thresholds
        for threshold in thresholds:
            pedestrians = filter_pedestrians(pedestrians, threshold)

        return pedestrians

    def get_groups(self, ids=[], days=None, thresholds=[], size=None) -> list[Group]:
        if days is None:
            days = self.days

        groups = []
        for day in days:
            traj_path = os.path.join(self.data_dir, f"trajectories_{day}.pkl")
            if not os.path.exists(traj_path):
                raise ValueError(
                    f"Coud not find trajectory data for day {day} in {self.data_dir}."
                )

            daily_traj = pickle_load(traj_path)

            groups_annotations_path = os.path.join(
                self.data_dir, f"groups_annotations_{day}.pkl"
            )
            if not os.path.exists(groups_annotations_path):
                raise ValueError(
                    f"Coud not find annotation data for day {day} in {self.data_dir}."
                )
            groups_annotations = pickle_load(groups_annotations_path)

            for group_id in groups_annotations:
                if ids and group_id not in ids:
                    continue
                group_data = groups_annotations[group_id]
                members = []
                # skipping groups of the wrong size
                if size is not None and group_data["size"] != size:
                    continue
                # if group_data["size"] != len(group_data["members"]):
                #     print("here")
                for group_member_id in group_data["members"]:
                    if group_member_id in daily_traj:
                        trajectory = daily_traj[group_member_id]
                    else:
                        continue
                    group_member = Pedestrian(
                        group_member_id, self, day, trajectory, [group_id]
                    )
                    members += [group_member]
                # apply the potential thresholds
                for threshold in thresholds:
                    members = filter_pedestrians(members, threshold)
                # missing
                if len(members) != group_data["size"]:
                    # print(
                    #     f"Skipping group {group_id}, trajectory missing for members."
                    # )
                    continue
                group = Group(group_id, members, self, day, group_data)
                groups += [group]

        return groups

    def get_groups_grouped_by(
        self, group_by_value, thresholds=[], days=None, size=None
    ):

        groups = self.get_groups(days, size=size, thresholds=thresholds)

        grouped_groups = {}
        for group in groups:
            if group_by_value not in group.annotations:
                # raise AttributeError(
                #     f"Group-by value '{group_by_value}' not found in group {group}."
                # )
                continue
            value = group.annotations[group_by_value]
            if value not in grouped_groups:
                grouped_groups[value] = []
            grouped_groups[value] += [group]

        return grouped_groups
