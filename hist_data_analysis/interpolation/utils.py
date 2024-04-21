import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json


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

def save_csv(data, filename):
    """
    Save data to a CSV file.

    :param data: dictionary
    :param filename: str
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
