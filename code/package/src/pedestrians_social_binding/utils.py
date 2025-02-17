import pickle as pk
import pandas as pd
from scipy import stats
import numpy as np


def pickle_load(file_path: str):
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


def pickle_save(file_path: str, data):
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


def write_table(data: list, label: list):
    cohen_data = np.zeros((len(data), len(data)))
    t_statistic = np.zeros((len(data), len(data)))
    for i, line in enumerate(data):
        for j in range(i + 1, len(data)):
            t_statistic[i, j] = stats.ttest_ind(line, data[j])[1]
            t_statistic[j, i] = None
            cohen_data[i, j] = np.abs(compute_cohen_d(line, data[j]))
            cohen_data[j, i] = None

    final_data_stat = {
        "Labels / Data ": label,
    }
    final_data_cohen = {
        "Labels / Data ": label,
    }

    for i, label_name in enumerate(label):
        final_data_stat[label_name] = t_statistic[i, :]
        final_data_cohen[label_name] = cohen_data[i, :]

    return [pd.DataFrame(final_data_stat), pd.DataFrame(final_data_cohen)]


def compute_cohen_d(group1: list, group2: list):
    n1 = len(group1)
    n2 = len(group2)
    return (np.mean(group1) - np.mean(group2)) / np.sqrt(
        ((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2)
    )
