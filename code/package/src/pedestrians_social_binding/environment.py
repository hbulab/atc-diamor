from pedestrians_social_binding.utils import *
from pedestrians_social_binding.trajectory_utils import *
from pedestrians_social_binding.constants import *
from pedestrians_social_binding.pedestrian import Pedestrian
from pedestrians_social_binding.group import Group

import os


class Environment:
    """Class representing an environment (ATC or DIAMOR). It contains the data of the pedestrians and groups.
    It is used to load the data and to get the pedestrians and groups.
    The data is loaded from the data directory.
    The data directory should contain the following files:
    - trajectories_{day}.pkl: a dictionary containing the trajectories of the pedestrians for each day
    - individuals_annotations_{day}.pkl: a dictionary containing the annotations of the pedestrians for each day
    - groups_annotations_{day}.pkl: a dictionary containing the annotations of the groups for each day

    ----------

    Constructor of the Environment class.

    Parameters
    - name: name of the environment. Can be "atc", "atc:corridor", "diamor" or "diamor:corridor".
    - data_dir: path to the directory containing the data.
    """

    def __init__(self, name, data_dir, raw=False):
        if name not in ["atc", "atc:corridor", "diamor", "diamor:corridor"]:
            raise ValueError(f"Unknown environment {name}.")
        self.short_name = name.split(":")[0]
        self.name = name
        self.data_dir = os.path.join(data_dir, self.short_name)
        self.boundaries = BOUNDARIES[name]
        self.raw = raw
        self.days = (
            DAYS_ATC
            if self.name == "atc" or self.name == "atc:corridor"
            else DAYS_DIAMOR
        )

    """Get the boundaries of the environment"""

    def get_boundaries(self):
        return [
            self.boundaries["xmin"],
            self.boundaries["xmax"],
            self.boundaries["ymin"],
            self.boundaries["ymax"],
        ]

    """Get the pedestrians of the environment
    - ids: list of ids of the pedestrians to get. If empty, all pedestrians are returned.
    - thresholds: list of thresholds to apply to the pedestrians. If empty, no threshold is applied.
    - no_groups: if True, only pedestrians that are not in a group are returned.
    - days: list of days to get the pedestrians from. If empty, all days are used.
    - sampling_time: if not None, the trajectories are resampled with the given sampling time.
    """

    def get_pedestrians(
        self, ids=[], thresholds=[], no_groups=False, days=None, sampling_time=None
    ) -> list[Pedestrian]:
        """Get the pedestrians of the environment

        Parameters
        ----------
        ids : list of int, optional, list of ids of the pedestrians to get. If empty, all pedestrians are returned.
        thresholds : list of float, optional, list of thresholds to apply to the pedestrians. If empty, no threshold is applied.
        no_groups : bool, optional, if True, only pedestrians that are not in a group are returned.
        days : list of int, optional, list of days to get the pedestrians from. If empty, all days are used.
        sampling_time : float, optional, if not None, the trajectories are resampled with the given sampling time.

        Returns
        -------
        list of Pedestrian, the list of pedestrians
        """

        if days is None:
            days = self.days

        pedestrians = []
        for day in days:
            if self.raw:
                traj_path = os.path.join(self.data_dir, f"trajectories_raw_{day}.pkl")
            else:
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
                if sampling_time is not None:
                    trajectory = resample_trajectory(
                        trajectory, sampling_time=sampling_time
                    )
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

    """Get the pedestrians of the environment grouped by a given value
    - group_by_value: the value to group the pedestrians by. It should be a property of the Pedestrian class.
    - ids: list of ids of the pedestrians to get. If empty, all pedestrians are returned.
    - thresholds: list of thresholds to apply to the pedestrians. If empty, no threshold is applied.
    - no_groups: if True, only pedestrians that are not in a group are returned.
    - days: list of days to get the pedestrians from. If empty, all days are used.
    - sampling_time: if not None, the trajectories are resampled with the given sampling time.
    """

    def get_pedestrians_grouped_by(
        self,
        group_by_value,
        ids=[],
        thresholds=[],
        no_groups=False,
        days=None,
        sampling_time=None,
    ) -> dict[any, list[Pedestrian]]:
        """Get the pedestrians of the environment grouped by a given value

        Parameters
        ----------

        group_by_value : any, the value to group the pedestrians by. It should be a property of the Pedestrian class.
        ids : list of int, optional, list of ids of the pedestrians to get. If empty, all pedestrians are returned.
        thresholds : list of float, optional, list of thresholds to apply to the pedestrians. If empty, no threshold is applied.
        no_groups : bool, optional, if True, only pedestrians that are not in a group are returned.
        days : list of int, optional, list of days to get the pedestrians from. If empty, all days are used.
        sampling_time : float, optional, if not None, the trajectories are resampled with the given sampling time.

        Returns
        -------
        dict of list of Pedestrian, the pedestrians grouped by the given value
        """
        pedestrians = self.get_pedestrians(
            ids=ids,
            thresholds=thresholds,
            no_groups=no_groups,
            days=days,
            sampling_time=sampling_time,
        )

        grouped_pedestrians = {}
        for pedestrian in pedestrians:
            if not hasattr(pedestrian, group_by_value):
                # raise AttributeError(
                #     f"Group-by value '{group_by_value}' not found in group {group}."
                # )
                continue
            value = getattr(pedestrian, group_by_value)
            if value not in grouped_pedestrians:
                grouped_pedestrians[value] = []
            grouped_pedestrians[value] += [pedestrian]

        return grouped_pedestrians

    """Get the groups of the environment
    - ids: list of ids of the groups to get. If empty, all groups are returned.
    - days: list of days to get the groups from. If empty, all days are used.
    - ped_thresholds: list of thresholds to apply to the pedestrians of the groups. If empty, no threshold is applied.
    - group_thresholds: list of thresholds to apply to the groups. If empty, no threshold is applied.
    - size: if not None, only groups with the given size are returned.
    - with_social_binding: if True, only groups with social binding are returned.
    - sampling_time: if not None, the trajectories are resampled with the given sampling time.
    """

    def get_groups(
        self,
        ids=[],
        days=None,
        ped_thresholds=[],
        group_thresholds=[],
        size=None,
        with_social_binding=False,
        sampling_time=None,
    ) -> list[Group]:
        """Get the groups of the environment

        Parameters
        ----------

        ids : list of int, optional, list of ids of the groups to get. If empty, all groups are returned.
        days : list of int, optional, list of days to get the groups from. If empty, all days are used.
        ped_thresholds : list of float, optional, list of thresholds to apply to the pedestrians of the groups. If empty, no threshold is applied.
        group_thresholds : list of float, optional, list of thresholds to apply to the groups. If empty, no threshold is applied.
        size : int, optional, if not None, only groups with the given size are returned.
        with_social_binding : bool, optional, if True, only groups with social binding are returned.
        sampling_time : float, optional, if not None, the trajectories are resampled with the given sampling time.

        Returns
        -------
        list of Group, the groups of the environment
        """
        if days is None:
            days = self.days

        groups = []
        for day in days:
            if self.raw:
                traj_path = os.path.join(self.data_dir, f"trajectories_raw_{day}.pkl")
            else:
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

                if (
                    with_social_binding
                    and not SOCIAL_BINDING[self.short_name] in group_data
                ):
                    continue

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
                    if sampling_time is not None:
                        trajectory = resample_trajectory(
                            trajectory, sampling_time=sampling_time
                        )
                    group_member = Pedestrian(
                        group_member_id, self, day, trajectory, [group_id]
                    )
                    members += [group_member]
                # apply the potential ped thresholds
                for threshold in ped_thresholds:
                    members = filter_pedestrians(members, threshold)
                # missing
                if len(members) != group_data["size"]:
                    # print(
                    #     f"Skipping group {group_id}, trajectory missing for members."
                    # )
                    continue

                # apply the potential group thresholds
                group = Group(group_id, members, self, day, group_data)
                keep = True
                for threshold in group_thresholds:
                    if not filter_group(group, threshold):
                        keep = False
                        continue
                if keep:
                    groups += [group]

        return groups

    """Get the groups of the environment grouped by a given value
    - group_by_value: the value to group the groups by. It should be a property of the Group class.
    - ped_thresholds: list of thresholds to apply to the pedestrians of the groups. If empty, no threshold is applied.
    - group_thresholds: list of thresholds to apply to the groups. If empty, no threshold is applied.
    - days: list of days to get the groups from. If empty, all days are used.
    - size: if not None, only groups with the given size are returned.
    - with_social_binding: if True, only groups with social binding are returned.
    - sampling_time: if not None, the trajectories are resampled with the given sampling time.
    """

    def get_groups_grouped_by(
        self,
        group_by_value,
        ped_thresholds=[],
        group_thresholds=[],
        days=None,
        size=None,
        with_social_binding=False,
        sampling_time=None,
    ) -> dict[any, list[Group]]:
        """Get the groups of the environment grouped by a given value

        Parameters
        ----------
        group_by_value : any, the value to group the groups by. It should be a property of the Group class.
        size : int, optional, if not None, only groups with the given size are returned.
        ped_thresholds : list of float, optional, list of thresholds to apply to the pedestrians of the groups. If empty, no threshold is applied.
        group_thresholds : list of float, optional, list of thresholds to apply to the groups. If empty, no threshold is applied.
        days : list of int, optional, list of days to get the groups from. If empty, all days are used.
        with_social_binding : bool, optional, if True, only groups with social binding are returned.
        sampling_time : float, optional, if not None, the trajectories are resampled with the given sampling time.

        Returns
        -------
        dict of list of Group, the groups of the environment grouped by the given value

        """

        groups = self.get_groups(
            days,
            size=size,
            ped_thresholds=ped_thresholds,
            group_thresholds=group_thresholds,
            with_social_binding=with_social_binding,
            sampling_time=sampling_time,
        )

        grouped_groups = {}
        for group in groups:
            if group_by_value in group.annotations:
                value = group.annotations[group_by_value]
            elif hasattr(group, group_by_value):
                value = getattr(group, group_by_value)
            else:
                continue
            if value not in grouped_groups:
                grouped_groups[value] = []
            grouped_groups[value] += [group]

        return grouped_groups
