import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import public_weather_installation.utils as utils

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


def load(df, normalize=True):
    """
    Loads and preprocesses data from a CSV file.

    :param df: the CSV file
    :param normalize: normalization flag
    :return: dataframe
    """
    df.sort_values(by='DATETIME', inplace=True)
    assert len(df) == 8

    empty_days = df.groupby(df['DATETIME'].dt.date).apply(lambda x: x.dropna(subset=params["X"], how='all').empty)
    df = df[~df['DATETIME'].dt.date.isin(empty_days[empty_days].index)]

    datetimes, periods = ['month', 'day', 'hour'], [12, 30, 24]

    for i, dtime in enumerate(datetimes):
        timestamps = df['DATETIME'].dt.__getattribute__(dtime)
        df[f'SIN_{dtime.upper()}'] = np.sin(np.pi * timestamps / periods[i])
        df[f'COS_{dtime.upper()}'] = np.cos(np.pi * timestamps / periods[i])

    stats = utils.load_json(filename='weather_predictions/communication_PLC/public_weather_installation/stats.json')

    if normalize:
        df = utils.normalize(df, stats, exclude=['DATETIME', 'FORECAST_DATETIME', 'SIN_MONTH', 'COS_MONTH', 'SIN_DAY',
                                                 'COS_DAY', 'SIN_HOUR', 'COS_HOUR'])

    return df


def prepare(df, phase):
    """
    Prepares the dataframe for training by filtering columns and saving to CSV.

    :param df: dataframe
    :param phase: str model phase (train or test)
    :return: dataframe
    """
    name = "./weather_predictions/communication_PLC/public_weather_installation/" + "df_" + phase + ".csv"

    for column, threshold in params["ignore"]:
        df = utils.filter(df, column=column, threshold=threshold) 

    df.set_index('DATETIME', inplace=True)
    df.to_csv(name)
    df = pd.read_csv(name, parse_dates=['DATETIME'], index_col='DATETIME')

    return df


class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, per_day=False):
        """
        Initializes a time series dataset.

        :param df: dataframe
        :param seq_len: length of the input sequence
        :param X: input features names
        :param t: time-related features names
        :param per_day: boolean
        """
        self.seq_len = seq_len
        self.per_day = per_day

        self.X = pd.concat([df[X], df[t]], axis=1)

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
    
        X = self.X.iloc[start_idx:end_idx].values

        mask_X = pd.isnull(X).astype(int)

        X = torch.FloatTensor(X)
        mask_X = torch.FloatTensor(mask_X)

        X = X.masked_fill(mask_X == 1, -2)

        seq_len = mask_X.size(0)
        mask_X_1d = torch.zeros(seq_len)

        for i in range(seq_len):
            if torch.any(mask_X[i] == 1):
                mask_X_1d[i] = 1

        return X, mask_X_1d
    
    @property
    def max_seq_id(self):
        return self.X.shape[0] - self.seq_len
    
    @property
    def num_seqs(self):
        return self.X.shape[0] // self.seq_len