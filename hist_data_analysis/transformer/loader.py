import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import logging
from hist_data_analysis import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

data = {
    "X": ["humidity", "pressure", "feels_like", "temp", "wind_speed", "rain_1h", "snow_1h", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"],
    #"X": ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"],
    "t": ["SIN_HOUR", "COS_HOUR", "SIN_DAY", "COS_DAY", "SIN_MONTH", "COS_MONTH"],
    #"t": [],
    "y": ["binned_Q_PVT"],
    "ignore": [] 
}

def load(path, parse_dates, normalize=True):
    """
    Loads and preprocesses data from a CSV file.

    :param path: path to the CSV file
    :param parse_dates: columns to parse as dates in the dataframe
    :param normalize: normalization flag
    :return: dataframe
    """
    df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    df.sort_values(by='DATETIME', inplace=True)

    logger.info("All data: {} rows".format(len(df)))

    empty_days = df.groupby(df['DATETIME'].dt.date).apply(lambda x: x.dropna(subset=data["X"], how='all').empty)
    df = df[~df['DATETIME'].dt.date.isin(empty_days[empty_days].index)]

    logger.info("Number of empty days: %d", empty_days.sum())
    logger.info("Number of empty data points: %d", 8 * empty_days.sum())
    logger.info("Data after dropping NAN days: {} rows".format(len(df)))

    datetimes, periods = ['month', 'day', 'hour'], [12, 30, 24]

    for i, dtime in enumerate(datetimes):
        timestamps = df['DATETIME'].dt.__getattribute__(dtime)
        df[f'SIN_{dtime.upper()}'] = np.sin(2*np.pi*timestamps/periods[i])
        df[f'COS_{dtime.upper()}'] = np.cos(2*np.pi*timestamps/periods[i])

    if os.path.exists('data/stats.json'):
        stats = utils.load_json(filename='data/stats.json')
    else:
        stats = utils.get_stats(df, path='data/')

    if normalize:
        df = utils.normalize(df, stats, exclude=['DATETIME', 'SIN_MONTH', 'COS_MONTH', 'SIN_DAY', 
                                                 'COS_DAY', 'SIN_HOUR', 'COS_HOUR', 'binned_Q_PVT'])

    nan_counts = df.isna().sum() / len(df) * 100
    logger.info("NaN counts for columns in X: %s", nan_counts)

    return df

def prepare(df, phase):
    """
    Prepares the dataframe for training by filtering columns and saving to CSV.

    :param df: dataframe
    :param phase: str model phase (train or test)
    :return: dataframe
    """
    name = "data/" + "df_" + phase + ".csv"

    for column, threshold in data["ignore"]:
        df = utils.filter(df, column=column, threshold=threshold) 

    df.set_index('DATETIME', inplace=True)
    df.to_csv(name)
    df = pd.read_csv(name, parse_dates=['DATETIME'], index_col='DATETIME')

    return df
    
class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y):
        """
        Initializes a time series dataset.

        :param df: dataframe
        :param seq_len: length of the input sequence
        :param X: input features names
        :param y: target variables names
        """
        self.seq_len = seq_len

        y_nan = df[y].isna().any(axis=1)
        df.loc[y_nan, :] = float('nan')

        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]

    def __len__(self):
        """
        :return: number of sequences that can be created from dataset X
        """
        return self.max_seq_id + 1
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: index of the sample
        :return: tuple containing input features sequence and target variables sequence
        """
        start_idx = idx
        end_idx = start_idx + self.seq_len
    
        X, y = self.X.iloc[start_idx:end_idx].values, self.y.iloc[start_idx:end_idx].values

        mask_X, mask_y = pd.isnull(X).astype(int), pd.isnull(y).astype(int)

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        mask_X, mask_y = torch.FloatTensor(mask_X), torch.FloatTensor(mask_y)

        X, y = X.masked_fill(mask_X == 1, 0), y.masked_fill(mask_y == 1, 0)

        """ # COLUMNS "rain_3h", "snow_3h" HAVE ONLY 0! (a.k.a the old NANS)
        column_to_check = 6
        column_name = self.X.columns[column_to_check]
        column_has_nonzero = torch.any(X[:, column_to_check].flatten() != 0.)

        if column_has_nonzero.item():
           print("Feature:", column_name)
           print("Column has non-zero values:", column_has_nonzero.item())
        """

        """ # NO COLUMN HAS NANS
        column_to_check = 10
        column_name = self.X.columns[column_to_check]
        column_has_nans = torch.isnan(X[:, column_to_check]).any()

        if column_has_nans.item():
           print("Feature:", column_name)
           print("Column has nan values:", column_has_nans.item())
        """

        mask_X_2d = torch.zeros(mask_X.size(0))
        for i in range(mask_X.size(0)):
            if torch.any(mask_X[i] == 1):
                mask_X_2d[i] = 1
            else:
                mask_X_2d[i] = 0

        return X, y, mask_X_2d
    
    @property
    def max_seq_id(self):
        return self.X.shape[0] - self.seq_len
    
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