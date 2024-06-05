import numpy as np

# value for the classification into the different relative position
REL_DIR_ANGLE_COS = np.cos(np.pi / 8)

# percentage of relative direction observation necessary to classify
REL_DIR_MIN_PERC = 0.9

# percentage of relative side observation necessary to classify
ENC_SIDE_MIN_PERC = 0.75

# percentage of direction observation necessary to classify
DIR_MIN_PERC = 0.9
