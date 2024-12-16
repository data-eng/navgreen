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
    "X": ["TEMPERATURE", "HUMIDITY", "WIND_SPEED", "WIND_DIRECTION", "SKY"],
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

    logger.info("All data: {} rows".format(len(df)))

    empty_days = df.groupby(df['DATETIME'].dt.date).apply(lambda x: x.dropna(subset=params["X"], how='all').empty)
    df = df[~df['DATETIME'].dt.date.isin(empty_days[empty_days].index)]

    logger.info("Number of empty days: %d", empty_days.sum())
    logger.info("Number of empty data points: %d", 8 * empty_days.sum())
    logger.info("Data after dropping NAN days: {} rows".format(len(df)))

    datetimes, periods = ['month', 'day', 'hour'], [12, 30, 24]

    for i, dtime in enumerate(datetimes):
        timestamps = df['DATETIME'].dt.__getattribute__(dtime)
        '''df[f'SIN_{dtime.upper()}'] = np.sin(2*np.pi*timestamps/periods[i])
        df[f'COS_{dtime.upper()}'] = np.cos(2*np.pi*timestamps/periods[i])'''

        df[f'SIN_{dtime.upper()}'] = np.sin(np.pi * timestamps / periods[i])
        df[f'COS_{dtime.upper()}'] = np.cos(np.pi * timestamps / periods[i])

    if os.path.exists('transformer/stats_new.json'):
        stats_new = utils.load_json(filename='transformer/stats_new.json')
    else:
        stats_new = utils.get_stats(df, path='transformer/', name='stats_new.json')

    # For the values that are identical within the two datasets, use the old statistics
    stats_init = utils.load_json(filename='transformer/stats.json')
    stats = stats_new.copy()
    stats['TEMPERATURE'] = stats_init['temp']
    stats['HUMIDITY'] = stats_init['humidity']
    stats['WIND_SPEED'] = stats_init['wind_speed']

    # In this loader we will not save any weights, we use the ones from the initial dataset

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
    name = "transformer/" + "df_" + phase + ".csv"

    for column, threshold in params["ignore"]:
        df = utils.filter(df, column=column, threshold=threshold) 

    df.set_index('DATETIME', inplace=True)
    df.to_csv(name)
    df = pd.read_csv(name, parse_dates=['DATETIME'], index_col='DATETIME')

    return df


class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y, per_day=False, tune=False):
        """
        Initializes a time series dataset.

        :param df: dataframe
        :param seq_len: length of the input sequence
        :param X: input features names
        :param t: time-related features names
        :param y: target variables names
        :param per_day: boolean
        """
        self.seq_len = seq_len
        self.per_day = per_day

        y_nan = df[y].isna().any(axis=1)
        df.loc[y_nan, :] = float('nan')

        if tune:
            X = ['HUMIDITY', 'WIND_DIRECTION', 'TEMPERATURE', 'TEMPERATURE', 'WIND_SPEED', 'SKY']

        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]

    def __len__(self):
        """
        :return: number of sequences that can be created from dataset X
        """
        if not self.per_day:
            return self.max_seq_id + 1
        else:
            return self.num_seqs
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: index of the sample
        :return: tuple containing input features sequence, target variables sequence and their respective masks
        """
        if not self.per_day:
            start_idx = idx
        else:
            start_idx = idx * self.seq_len

        end_idx = start_idx + self.seq_len
    
        X, y = self.X.iloc[start_idx:end_idx].values, self.y.iloc[start_idx:end_idx].values

        mask_X, mask_y = pd.isnull(X).astype(int), pd.isnull(y).astype(int)

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        mask_X, mask_y = torch.FloatTensor(mask_X), torch.FloatTensor(mask_y)

        X, y = X.masked_fill(mask_X == 1, -2), y.masked_fill(mask_y == 1, 0)

        seq_len = mask_X.size(0)
        mask_X_1d = torch.zeros(seq_len)
        mask_y_1d = torch.zeros(seq_len)

        for i in range(seq_len):
            if torch.any(mask_X[i] == 1):
                mask_X_1d[i] = 1
            if torch.any(mask_y[i] == 1):
                mask_y_1d[i] = 1

        return X, y, mask_X_1d, mask_y_1d
    
    @property
    def max_seq_id(self):
        return self.X.shape[0] - self.seq_len
    
    @property
    def num_seqs(self):
        return self.X.shape[0] // self.seq_len


def split(dataset, vperc=0.2):
    """
    Splits a dataset into training and validation sets.

    :param dataset: dataset
    :param vperc: percentage of data to allocate for validation
    :return: tuple containing training and validation datasets
    """
    ds_seqs = int(len(dataset))

    valid_seqs = int(vperc * ds_seqs)
    train_seqs = ds_seqs - valid_seqs

    return random_split(dataset, [train_seqs, valid_seqs])
