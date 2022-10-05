# ---------- ATC constants ----------
DAYS_ATC = ["0109", "0217", "0424", "0505", "0508"]
SOCIAL_RELATIONS_JP = ["idontknow", "koibito", "doryo", "kazoku", "yuujin"]
SOCIAL_RELATIONS_EN = ["idontknow", "Couples", "Colleagues", "Families", "Friends"]
BOUNDARIES_ATC = {"xmin": -41000, "xmax": 49000, "ymin": -27500, "ymax": 24000}
BOUNDARIES_ATC_CORRIDOR = {"xmin": 5000, "xmax": 48000, "ymin": -27000, "ymax": 8000}


# ---------- DIAMOR constants ----------
DAYS_DIAMOR = ["06", "08"]
INTENSITIES_OF_INTERACTION_NUM = ["0", "1", "2", "3"]
BOUNDARIES_DIAMOR = {"xmin": -200, "xmax": 60300, "ymin": -5300, "ymax": 12000}
BOUNDARIES_DIAMOR_CORRIDOR = {
    "xmin": 20000,
    "xmax": 60300,
    "ymin": -5300,
    "ymax": 12000,
}


# ---------- other constants ----------
BOUNDARIES = {
    "atc": BOUNDARIES_ATC,
    "atc:corridor": BOUNDARIES_ATC_CORRIDOR,
    "diamor": BOUNDARIES_DIAMOR,
    "diamor:corridor": BOUNDARIES_DIAMOR_CORRIDOR,
}

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

THRESHOLDS = ["t", "x", "y", "z", "v", "v_x", "v_y", "d", "delta", "theta", "n"]

REL_DIRS = ["opposite", "same", "cross"]

DEFLECTION_MEASURES = [
    "straightness_index",
    "sinuosity",
    "maximum_scaled_lateral_deviation",
]

ALL_DEFLECTION_MEASURES = [
    "straightness_index",
    "sinuosity",
    "maximum_scaled_lateral_deviation",
    "maximum_lateral_deviation",
    "area_under_curve",
    "scaled_area_under_curve"
]

SOCIAL_BINDING = {"atc": "soc_rel", "diamor": "interaction"}
