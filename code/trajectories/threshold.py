from trajectories.constants import *


class Threshold:
    def __init__(self, value, min, max, aggregate=False):

        if value not in TRAJECTORY_COLUMNS:
            raise ValueError(
                f"Unknown threshold value {value}. Should be one of {TRAJECTORY_COLUMNS.keys()}"
            )

        self.value = value
        self.min = min
        self.max = max
        self.aggregate = aggregate

    def get_value(self):
        return self.value

    def get_min_value(self):
        return self.min

    def get_max_value(self):
        return self.max

    def is_aggregate(self):
        return self.aggregate
