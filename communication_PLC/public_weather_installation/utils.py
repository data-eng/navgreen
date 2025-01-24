import torch
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
import json
import math
from collections import namedtuple


def tensor_to_python_numbers(tensor):
    """
    Converts all items in a dictionary to numpy numbers
    :param tensor:
    :return:
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.item() if tensor.numel() == 1 else tensor.cpu().numpy().tolist()
    elif isinstance(tensor, np.ndarray):
        return tensor.item() if np.prod(tensor.shape) == 1 else tensor.tolist()
    elif isinstance(tensor, (list, tuple)):
        return [tensor_to_python_numbers(item) for item in tensor]
    elif isinstance(tensor, dict):
        return {key: tensor_to_python_numbers(value) for key, value in tensor.items()}
    else:
        return tensor


def mask(tensor, mask, id=0):
    """
    Mask a tensor based on a specified condition.

    :param tensor: torch.Tensor
    :param mask: torch.Tensor
    :param id: int value specifying the elements to keep
    :return: torch.Tensor
    """
    return tensor[mask == id]


def get_max(arr):
    """
    Get the maximum value and its index from an array.

    :param arr: numpy array
    :return: namedtuple
    """
    Info = namedtuple('Info', ['value', 'index'])

    max_index = np.argmax(arr)
    max_value = arr[max_index]

    return Info(value=max_value, index=max_index)


def hot3D(t, k, device):
    """
    Encode 3D tensor into one-hot format.

    :param t: tensor of shape (dim_0, dim_1, dim_2)
    :param k: int number of classes
    :param device: device
    :return: tensor of shape (dim_0, dim_1, k)
    """
    dim_0, dim_1, _ = t.size()
    t_hot = torch.zeros(dim_0, dim_1, k, device=device)

    for x in range(dim_0):
        for y in range(dim_1):
            for z in t[x, y]:
                t_hot[x, y] = torch.tensor(one_hot(z.item(), k=k))

    return t_hot.to(device)


def one_hot(val, k):
    """
    Convert categorical value to one-hot encoded representation.

    :param val: float
    :param k: number of classes
    :return: list
    """
    encoding = []

    if math.isnan(val):
        encoding = [val for _ in range(k)]
    else:
        encoding = [0 for _ in range(k)]
        encoding[int(val)] = 1

    return encoding


def get_prfs(true, pred, avg=['micro', 'macro', 'weighted'], include_support=False):
    """
    Calculate precision, recall, fscore and support using the given averaging methods.

    :param true: list
    :param pred: list
    :param avg: list
    :param include_support: boolean
    :return: dict
    """
    prfs = {}

    for method in avg:
        precision, recall, fscore, support = precision_recall_fscore_support(true, pred, average=method)

        prfs[f'precision_{method}'] = precision
        prfs[f'recall_{method}'] = recall
        prfs[f'fscore_{method}'] = fscore

        if include_support:
            prfs[f'support_{method}'] = support

    return prfs


def get_path(dirs, name=""):
    """
    Get the path by joining directory names.
    :param dirs: list
    :param name: name of the path
    :return: the path
    """
    dir_path = os.path.join(*dirs)
    os.makedirs(dir_path, exist_ok=True)

    return os.path.join(dir_path, name)


def save_json(data, filename):
    """
    Save data to a JSON file.
    :param data: dictionary
    :param filename: str
    """
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    """
    Load data from a JSON file.
    :param filename: str
    :return: dictionary
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    return data


def save_csv(data, filename, append=False):
    """
    Save data to a CSV file.

    :param data: dictionary
    :param filename: str
    """
    df = pd.DataFrame(data)
    if append:
        df.to_csv(filename, index=False, mode='a', header=False)
    else:
        df.to_csv(filename, index=False)



def normalize(df, stats, exclude=[]):
    """
    Normalize data.

    :param df: dataframe
    :param stats: tuple of mean and std
    :param exclude: column to exclude from normalization
    :return: processed dataframe
    """
    newdf = df.copy()

    for col in df.columns:
        if col not in exclude:
            series = df[col]
            mean, std = stats[col]
            series = (series - mean) / std
            newdf[col] = series

    return newdf


def get_stats(df, path='data/'):
    """
    Compute mean and standard deviation for each column in the dataframe.

    :param df: dataframe
    :return: dictionary
    """
    stats = {}
    filename = os.path.join(path, 'stats.pkl')

    for col in df.columns:
        if col != "DATETIME":
            series = df[col]
            mean = series.mean()
            std = series.std()
            stats[col] = (mean, std)

    filename = os.path.join(path, 'stats.json')
    save_json(data=stats, filename=filename)

    return stats


def filter(df, column, threshold):
    """
    Filter dataframe based on a single column and its threshold if the column exists.

    :param df: dataframe
    :param column: column name to filter
    :param threshold: threshold value for filtering
    :return: filtered dataframe if column exists, otherwise original dataframe
    """
    if column in df.columns:
        if threshold is not None:
            df = df[df[column] > threshold]
        else:
            df.drop(column, axis="columns", inplace=True)

    return df


def aggregate(df, grp="1min", func=lambda x: x):
    """
    Resample dataframe based on the provided frequency and aggregate using the specified function.

    :param df: dataframe
    :param grp: resampling frequency ('1min' -> original)
    :param func: aggregation function (lambda x: x -> no aggregation)
    :return: aggregated dataframe
    """
    df = df.set_index("DATETIME")

    if grp:
        df = df.resample(grp)
        df = df.apply(func)
        df = df.dropna()

    df = df.sort_index()

    return df
