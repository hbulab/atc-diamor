from .utils import *
from .constants import *
from .pedestrian import Pedestrian

import os


class Environment:
    def __init__(self, name, data_dir):

        if name not in ["atc", "diamor"]:
            raise ValueError(f"Unknown environment {name}.")
        self.name = name

        self.data_dir = os.path.join(data_dir, self.name)

        self.boundaries = BOUNDARIES_ATC if self.name == "atc" else BOUNDARIES_DIAMOR

    def get_pedestrians(self, days=None):

        if days is None:
            days = DAYS_ATC if self.name == "atc" else DAYS_DIAMOR

        trajectories = []
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
                trajectories += [pedestrian]

        return trajectories
