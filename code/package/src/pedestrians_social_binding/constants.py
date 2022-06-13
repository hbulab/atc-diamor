# ---------- ATC constants ----------
DAYS_ATC = ["0109", "0217", "0424", "0505", "0508"]
SOCIAL_RELATIONS_JP = ["idontknow", "koibito", "doryo", "kazoku", "yuujin"]
SOCIAL_RELATIONS_EN = ["idontknow", "couples", "colleagues", "family", "friends"]
BOUNDARIES_ATC = {"xmin": -41000, "xmax": 49000, "ymin": -27500, "ymax": 24000}

# ---------- DIAMOR constants ----------
DAYS_DIAMOR = ["06", "08"]
BOUNDARIES_DIAMOR = {"xmin": -200, "xmax": 60300, "ymin": -5300, "ymax": 12000}
INTENSITIES_OF_INTERACTION_NUM = ["0", "1", "2", "3"]


COLORS = [
    "steelblue",
    "yellowgreen",
    "lightcoral",
    "orange",
    "gold",
    "rebeccapurple",
    "crimson",
]

TRAJECTORY_COLUMNS = {
    "t": 0,
    "x": 1,
    "y": 2,
    "z": 3,
    "v": 4,
    "v_x": 5,
    "v_y": 6,
}

THRESHOLDS = ["t", "x", "y", "z", "v", "v_x", "v_y", "d", "delta"]

REL_DIRS = ["opposite", "same", "cross"]

DEFLECTION_MEASURES = [
    "straightness_index",
    "sinuosity",
    "maximum_scaled_lateral_deviation",
]
