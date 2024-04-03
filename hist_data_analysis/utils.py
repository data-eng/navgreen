import os
import json
import scipy.signal
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as sched

def visualize(values, labels, title, names, colors, plot_func):
    """
    Visualize multiple (x,y) data.

    :param values: list of tuples
    :param labels: tuple
    :param title: str
    :param names: list of str
    :param colors: list of str
    :param plot_func: plotting function(e.g., plt.scatter, plt.plot)
    """
    x_label, y_label = labels

    plt.figure(figsize=(10, 6))
    for i, (x_values, y_values) in enumerate(values):
        plot_func(x_values, y_values, color=colors[i], label=names[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title="Parameters")
    plt.tight_layout()

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join('static/', filename))
    plt.close()

def normalize(df, stats):
    """
    Normalize and de-trend data.

    :param df: dataframe
    :param stats: tuple of mean and std
    :return: processed dataframe
    """
    newdf = df.copy()
    for col in df.columns:
        if col != "DATETIME":
            series = df[col]
            mean, std = stats[col]
            series = (series - mean) / std
            series = scipy.signal.detrend(series)
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