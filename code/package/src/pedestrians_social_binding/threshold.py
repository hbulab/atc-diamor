from pedestrians_social_binding.constants import *


class Threshold:
    """Class representing a threshold

    Attributes
    ----------
    value : str, the threshold value
    min : int, the minimum value of the threshold
    max : int, the maximum value of the threshold

    Constructor
    -----------
    value : str, the threshold value
    min : int, the minimum value of the threshold
    max : int, the maximum value of the threshold

    """
    def __init__(self, value, min=None, max=None):

        if value not in THRESHOLDS:
            raise ValueError(
                f"Unknown threshold value {value}. Should be one of {THRESHOLDS}"
            )

        self.value = value
        self.min = min
        self.max = max

    def get_value(self):
        """Get the threshold value
        """
        return self.value

    def get_min_value(self):
        """Get the threshold minimum value"""
        return self.min

    def get_max_value(self):
        """Get the threshold maximum value"""
        return self.max
