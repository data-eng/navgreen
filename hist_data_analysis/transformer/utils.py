import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import namedtuple
import torch.optim.lr_scheduler as sched

def mask(tensor, mask, id=0):
    """
    Mask a tensor based on a specified condition.

    :param tensor: torch.Tensor
    :param mask: torch.Tensor
    :param id: int value specifying the elements to keep
    :return: torch.Tensor
    """
    return tensor[mask == id]

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        """
        Initialize the WeightedCrossEntropyLoss module.

        :param weights: dictionary
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = self.extract_weights(weights)

    def extract_weights(self, weights):
        """
        Extract weights from the given dictionary and convert them to a tensor.

        :param weights: dictionary
        :return: tensor
        """
        weights = dict(sorted(weights.items()))
        weights = [weights[str(i)] for i in range(len(weights))]
        return torch.tensor(weights)
    
    def forward(self, pred, true):
        """
        Compute the weighted cross-entropy loss.

        :param pred: tensor (batch_size * seq_len, num_classes)
        :param true: tensor (batch_size * seq_len)
        :return: tensor
        """
        if true.size(0) == 0 or pred.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        loss = F.cross_entropy(pred, true, weight=self.weights.to(pred.device))

        return loss
    
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

def visualize(type, values, labels, title, plot_func=None, coloring=None, names=None, classes=None, tick=False, path=''):
    """
    Visualize (x,y) data points.

    :param type: str
    :param values: list of tuples / tuple
    :param labels: tuple
    :param title: str
    :param plot_func: plotting function (optional)
    :param colors: list / str (optional)
    :param names: list (optional)
    :param tick: bool (optional)
    :param classes: list (optional)
    """
    x_label, y_label = labels
    plt.figure(figsize=(10, 6))

    if type == 'single-plot':
        x_values, y_values = values
        plot_func(x_values, y_values, color=coloring)
        if tick:
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)

    elif type == 'multi-plot':
        x_values, y_values = values
        for i, (x_values, y_values) in enumerate(values):
            plot_func(x_values, y_values, color=coloring[i], label=names[i])
        if tick:
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)

    elif type == 'heatmap':
        x_values, y_values = values
        cm = confusion_matrix(x_values, y_values)
        cmap = sns.blend_palette(coloring, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(path, filename))
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

def get_path(dirs, name=""):
    """
    Get the path by joining directory names.

    :param dirs: list
    :param name: str
    :return: str
    """
    dir_path = os.path.join(*dirs)
    os.makedirs(dir_path, exist_ok=True)

    return os.path.join(dir_path, name)

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

def save_csv(data, filename):
    """
    Save data to a CSV file.

    :param data: dictionary
    :param filename: str
    """
    df = pd.DataFrame(data)      
    df.to_csv(filename, index=False)

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