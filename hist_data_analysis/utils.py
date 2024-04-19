import os
import json
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple
import torch.optim.lr_scheduler as sched

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

def visualize(values, labels, title, color, plot_func):
    """
    Visualize (x,y) data points.

    :param values: tuple of lists
    :param labels: tuple of str
    :param title: str
    :param color: str
    :param plot_func: plotting function
    """
    x_values, y_values = values
    x_label, y_label = labels

    plt.figure(figsize=(10, 6))
    plot_func(x_values, y_values, color=color)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join('static/', filename))
    plt.close()

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

def load_json(filename):
    """
    Load data from a JSON file.

    :param filename: str
    :return: dictionary
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    return data

def save_json(data, filename):
    """
    Save data to a JSON file.

    :param data: dictionary
    :param filename: str
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

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

def get_optim(name, model, lr):
    """
    Get optimizer object based on name, model, and learning rate.

    :param name: str
    :param model: model
    :param lr: float
    :return: optimizer object
    """
    optim_class = getattr(optim, name)
    optimizer = optim_class(model.parameters(), lr=lr)

    return optimizer

def get_sched(name, step_size, gamma, optimizer):
    """
    Get scheduler object based on name, step size, gamma, and optimizer.

    :param name: str
    :param step_size: int
    :param gamma: gamma float
    :param optimizer: optimizer object
    :return: scheduler object
    """
    sched_class = getattr(sched, name)
    scheduler = sched_class(optimizer, step_size, gamma)

    return scheduler