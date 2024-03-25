import scipy.signal
import pandas as pd
import torch.optim as optim
import torch.optim.lr_scheduler as sched

def normalize(df):
    """
    Normalize and de-trend data
    :param df: dataframe
    :return: processed dataframe
    """
    newdf = df.copy()
    for col in df.columns:
        if col != "DATETIME":
            series = df[col]
            series = (series - series.mean()) / series.std()
            series = scipy.signal.detrend(series)
            newdf[col] = series
    return newdf

def filter(df, column, threshold):
    """
    Filter dataframe based on a single column and its threshold if the column exists
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
    optim_class = getattr(optim, name)
    optimizer = optim_class(model.parameters(), lr=lr)
    return optimizer

def get_sched(name, step_size, gamma, optimizer):
    sched_class = getattr(sched, name)
    scheduler = sched_class(optimizer, step_size, gamma)
    return scheduler