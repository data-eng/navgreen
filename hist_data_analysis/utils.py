import scipy.signal
import pandas as pd

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

def filter(df, columns, thresholds):
    """
    Filter dataframe based on multiple columns and their respective thresholds if the columns exist
    :param df: dataframe
    :param columns: list of column names to filter
    :param thresholds: list of threshold values for filtering
    :return: filtered dataframe if columns exist, otherwise original dataframe
    """
    for col, threshold in zip(columns, thresholds):
        if col in df.columns:
            if threshold is not None:
                df = df[df[col] > threshold]
            else:
                df.drop(col, axis="columns", inplace=True)
    return df

def aggregate(df, grp, func):
    """
    Resample dataframe based on the provided frequency and aggregate using the specified function.
    
    :param df: dataframe
    :param grp: resampling frequency
    :param func: aggregation function
    :return: aggregated dataframe
    """
    df = df.set_index("DATETIME")
    if grp:
        df = df.resample(grp)
        df = df.apply(func)
        df = df.dropna()
    df = df.sort_index()
    return df