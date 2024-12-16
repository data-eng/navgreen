import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import logging
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

params = {
    "X": ["humidity", "pressure", "feels_like", "temp", "wind_speed", "rain_1h"],
    "t": ["SIN_HOUR", "COS_HOUR", "SIN_DAY", "COS_DAY", "SIN_MONTH", "COS_MONTH"],
    "ignore": [] 
}

def load(path, parse_dates, bin, normalize=True):
    """
    Loads and preprocesses data from a CSV file.

    :param path: path to the CSV file
    :param parse_dates: columns to parse as dates in the dataframe
    :param normalize: normalization flag
    :param bin: y_bin
    :return: dataframe
    """
    df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    df.sort_values(by='DATETIME', inplace=True)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    empty_days = df.groupby(df['DATETIME'].dt.date).apply(lambda x: x.dropna(subset=params["X"], how='all').empty)
    df = df[~df['DATETIME'].dt.date.isin(empty_days[empty_days].index)]

    datetimes, periods = ['month', 'day', 'hour'], [12, 30, 24]

    for i, dtime in enumerate(datetimes):
        timestamps = df['DATETIME'].dt.__getattribute__(dtime)
        df[f'SIN_{dtime.upper()}'] = np.sin(2*np.pi*timestamps/periods[i])
        df[f'COS_{dtime.upper()}'] = np.cos(2*np.pi*timestamps/periods[i])

    if os.path.exists('transformer/stats.json'):
        stats = utils.load_json(filename='transformer/stats.json')
    else:
        stats = utils.get_stats(df, path='transformer/')

    occs = df[bin].value_counts().to_dict()
    freqs = {int(key): value / sum(occs.values()) for key, value in occs.items()}
    utils.save_json(data=freqs, filename=f'transformer/freqs_{bin}.json')

    inverse_occs = {int(key): 1 / value for key, value in occs.items()}
    weights = {key: value / sum(inverse_occs.values()) for key, value in inverse_occs.items()}

    if not os.path.exists(f'transformer/weights_{bin}.json'):
        utils.save_json(data=weights, filename=f'transformer/weights_{bin}.json')

    if normalize:
        df = utils.normalize(df, stats, exclude=['DATETIME', 'SIN_MONTH', 'COS_MONTH', 'SIN_DAY', 
                                                 'COS_DAY', 'SIN_HOUR', 'COS_HOUR', bin])

    return df


def prepare(df, phase):
    """
    Prepares the dataframe for training by filtering columns and saving to CSV.

    :param df: dataframe
    :param phase: str model phase (train or test)
    :return: dataframe
    """
    name = "transformer/" + "df_" + phase + "_init.csv"

    for column, threshold in params["ignore"]:
        df = utils.filter(df, column=column, threshold=threshold) 

    df.set_index('DATETIME', inplace=True)
    df.to_csv(name)
    df = pd.read_csv(name, parse_dates=['DATETIME'], index_col='DATETIME')

    return df

