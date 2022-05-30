import numpy as np


def parse_group_data(group_data):

    ped_id = group_data[0]
    group_size = group_data[1]

    track_type = "normal"
    if ped_id == -1:  # not tracked pedestrian
        track_type = "missing"

    elif ped_id < 0:  # non pedestrian
        if group_size == -2:
            track_type = "wheelchair"
        elif group_size == -3:
            track_type = "automatic_wheelchair"
        else:
            # print(group_data)
            track_type = "baby_carriage"

    group_members_ids = []
    interacting_ids = []
    if group_size > 0:
        group_members_ids = group_data[2 : 2 + group_size - 1]
        n_interacting = group_data[2 + group_size - 1]

        interacting_ids = group_data[2 + group_size : 2 + group_size + n_interacting]

    return track_type, ped_id, group_members_ids, interacting_ids
