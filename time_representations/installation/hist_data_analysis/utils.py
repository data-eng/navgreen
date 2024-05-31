import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import math
from collections import namedtuple


class CrossEntropyLoss(nn.Module):
    """
    Custom cross entropy loss that considers masks
    """
    def __init__(self, weights):
        super(CrossEntropyLoss, self).__init__()
        w = 1.0 / weights
        self.weights = w / w.sum()
        #print(f'weights is {self.weights}')

    def forward(self, pred, true, mask):
        if pred.dim() == 2: pred = pred.permute(1, 0).unsqueeze(0)
        true = true.long()
        true = true * mask.long()
        loss = [F.cross_entropy(pred[b_sz, :, :], true[b_sz, :], reduction='none', weight=self.weights) for b_sz in range(true.shape[0])]
        loss = torch.stack(loss, dim=0)
        mask = mask.float()
        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss


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

class MaskedMSELoss(nn.Module):
    """
    MaskedMSELoss that utilizes masks
    """
    def __init__(self, sequence_length):
        super(MaskedMSELoss, self).__init__()
        self.sequence_length = sequence_length

    def forward(self, pred, true, mask):
        if pred.dim() == 1: pred = pred.unsqueeze(0)
        pred = pred.view(pred.shape[0], self.sequence_length, pred.shape[1] // self.sequence_length).mean(dim=2)
        # Compute element-wise squared difference
        squared_diff = (pred - true) ** 2
        # Apply mask to ignore certain elements
        mask = mask.float()
        loss = squared_diff * mask
        # Compute the mean loss only over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedSmoothL1Loss(nn.Module):
    """
    SmoothL1Loss that utilizes masks
    """
    def __init__(self, sequence_length):
        super(MaskedSmoothL1Loss, self).__init__()
        self.sequence_length = sequence_length

    def forward(self, pred, true, mask):
        '''
        if pred.dim() == 1: pred = pred.unsqueeze(0)
        pred = pred.view(pred.shape[0], self.sequence_length, pred.shape[1] // self.sequence_length).mean(dim=2)
        '''
        # Compute element-wise absolute difference
        abs_diff = torch.abs(pred - true)
        # Apply mask to ignore certain elements
        mask = mask.float()
        abs_diff = abs_diff * mask
        # Compute loss
        loss = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        # Compute the mean loss only over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero

        return loss


class MaskedCrossEntropyLoss_mTAN(nn.Module):
    """
    Cross Entropy Loss that utilizes masks
    """
    def __init__(self, sequence_length, weights):
        super(MaskedCrossEntropyLoss_mTAN, self).__init__()
        self.sequence_length = sequence_length
        # Calculate the inverse class frequencies
        if weights is not None:
            w = 1.0 / weights
            self.weights = w / w.sum()
        else: self.weights = None
        #print(f'weights is {self.weights}')

    def forward(self, pred, true, mask):
        if pred.dim() == 2: pred = pred.permute(1, 0).unsqueeze(0)

        #pred = pred.view(pred.shape[0], self.sequence_length, pred.shape[1] // self.sequence_length,
        #                  pred.shape[2]).mean(dim=2)

        true = true[mask == 1].long()
        pred = pred[mask == 1]

        loss = F.cross_entropy(pred, true, weight=self.weights)

        return loss


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

def save_csv(data, filename):
    """
    Save data to a CSV file.

    :param data: dictionary
    :param filename: str
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

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
            plt.legend()
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
    plt.savefig(os.path.join(path, filename), dpi=300)
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

def class_wise_pr_roc(labels, predicted_labels, name):
    num_classes = len(np.unique(labels))
    predicted_probs = label_binarize(predicted_labels, classes=np.arange(num_classes))
    precision = dict()
    recall = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pr_auc = dict()

    for i in range(num_classes):
        # Compute precision and recall for each class
        precision[i], recall[i], _ = precision_recall_curve(labels == i, predicted_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

        # Compute ROC curve for each class
        fpr[i], tpr[i], _ = roc_curve(labels == i, predicted_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot PR curve
        plt.figure()
        plt.plot(recall[i], precision[i], lw=2, label='PR curve (area = %0.2f)' % pr_auc[i])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve for class {i}')
        plt.legend(loc="lower left")
        plt.savefig(f'figures/pr_class_{i}_{name}.png', dpi=300)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for class {i}')
        plt.legend(loc="lower right")
        plt.savefig(f'figures/roc_class_{i}_{name}.png', dpi=300)


