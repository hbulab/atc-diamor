from pedestrians_social_binding.constants import *


class Threshold:
    def __init__(self, value, min=None, max=None):

        if value not in THRESHOLDS:
            raise ValueError(
                f"Unknown threshold value {value}. Should be one of {THRESHOLDS}"
            )

        self.value = value
        self.min = min
        self.max = max

    def get_value(self):
        return self.value

    def get_min_value(self):
        return self.min

    def get_max_value(self):
        return self.max
