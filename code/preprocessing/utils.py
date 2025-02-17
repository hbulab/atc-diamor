import numpy as np
import pickle as pk


def pickle_load(file_path):
    """Load the content of a pickle file

    Parameters
    ----------
    file_path : str
        The path to the file which will be unpickled

    Returns
    -------
    obj
        The content of the pickle file
    """
    with open(file_path, "rb") as f:
        data = pk.load(f)
    return data


def pickle_save(file_path, data):
    """Save data to pickle file

    Parameters
    ----------
    file_path : str
        The path to the file where the data will be saved
    data : obj
        The data to save
    """
    with open(file_path, "wb") as f:
        pk.dump(data, f)


def dat_load(file_path):
    """Load the content of a dat file

    Parameters
    ----------
    file_path : str
        The path to the file which will be read

    Returns
    -------
    obj
        The content of the dat file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(list(map(int, line.strip().split())))
    return data
