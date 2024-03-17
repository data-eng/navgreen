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