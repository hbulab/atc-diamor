from .utils import *
from .constants import *
from .pedestrian import Pedestrian
from .group import Group

import os


class Environment:
    def __init__(self, name, data_dir):

        if name not in ["atc", "diamor"]:
            raise ValueError(f"Unknown environment {name}.")
        self.name = name
        self.data_dir = os.path.join(data_dir, self.name)
        self.boundaries = BOUNDARIES_ATC if self.name == "atc" else BOUNDARIES_DIAMOR
        self.days = DAYS_ATC if self.name == "atc" else DAYS_DIAMOR

    def get_pedestrians(self, days=None):

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
                trajectory = daily_traj[ped_id]
                if ped_id in individual_annotations:
                    groups = individual_annotations[ped_id]["groups"]
                else:
                    groups = []

                pedestrian = Pedestrian(ped_id, self, day, trajectory, groups)
                pedestrians += [pedestrian]

        return pedestrians

    def get_groups(self, days=None, size=None):
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
                group_data = groups_annotations[group_id]
                members = []
                # skipping groups of the wrong size
                if size is not None and group_data["size"] != size:
                    continue
                for group_member_id in group_data["members"]:
                    if group_member_id in daily_traj:
                        trajectory = daily_traj[group_member_id]
                    else:
                        # print(
                        #     f"Skipping group {group_id}, trajectory missing for {group_member_id}."
                        # )
                        continue
                    group_member = Pedestrian(
                        group_member_id, self, day, trajectory, [group_id]
                    )
                    members += [group_member]
                group = Group(group_id, members, self, day)
                groups += [group]

        return groups
