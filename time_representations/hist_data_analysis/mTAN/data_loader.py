import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import logging

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
# Configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(stream_handler)


def load_df(df_path, pvt_cols, y_cols, parse_dates, normalize=True, stats=None):
    """
    Loads the data from the historical data DataFrame
    :return: whole dataframe, dataframe for hp, dataframe for solar
    """
    df = pd.read_csv(df_path, parse_dates=parse_dates, low_memory=False)

    # Drop unneeded columns
    for c in df.columns:
        if c not in pvt_cols:
            df.drop(c, axis="columns", inplace=True)

    mean_stds = {} if stats is None else stats

    # Normalize each column
    if normalize:
        for c in [c for c in df.columns if c != "DATETIME"]:
            if c not in y_cols:
                series = df[c]
                if stats is None: mean_stds[c] = (series.mean(), series.std())
                df[c] = (series - mean_stds[c][0]) / mean_stds[c][1]

    if stats is not None: mean_stds = None

    return df, mean_stds


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length, X_cols, y_cols, final_train=False, per_day=False):
        self.sequence_length = sequence_length
        self.per_day = per_day

        df = dataframe
        df['Datetime'] = df.index.astype('int64')
        # Normalize Unix time between 0 and 1
        min_time, max_time = df['Datetime'].min(), df['Datetime'].max()
        df['Datetime'] = (df['Datetime'] - min_time) / (max_time - min_time)

        # Check if any column in y_cols contains NaN values for each row
        has_nan_in_y_cols = df[y_cols].isna().any(axis=1)
        # Replace rows with NaN values in any of the y_cols with NaN values
        df.loc[has_nan_in_y_cols, 'Datetime'] = float('nan')
        #print(f'total {df.shape} -> nan {has_nan_in_y_cols.value_counts()}')
        self.X, self.y = df[X_cols], df[y_cols]
        self.time = df['Datetime']

        if final_train:
            for col in y_cols:
                print(f"Column {col} : min is {self.y[col].dropna().min():.4f} and max is {self.y[col].dropna().max():.4f}")
                print( f"Column {col} : mean is {self.y[col].dropna().mean():.4f} and std is {self.y[col].dropna().std():.4f}")

    def __len__(self):
        if not self.per_day: return self.X.shape[0] - self.sequence_length + 1
        else:
            assert self.X.shape[0] % self.sequence_length == 0
            return self.X.shape[0] // self.sequence_length

    def __getitem__(self, idx):
        if not self.per_day: start_idx = idx
        else: start_idx = idx * self.sequence_length

        end_idx = start_idx + self.sequence_length
        X = self.X.iloc[start_idx:end_idx].values
        y = self.y.iloc[start_idx:end_idx].values.flatten()
        time = self.time.iloc[start_idx:end_idx].values
        time = np.nan_to_num(time, nan=0)

        # Convert the sequence to a PyTorch tensor
        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        time = torch.FloatTensor(time)

        masks_X = torch.isnan(X).int()
        masks_X = 1 - masks_X
        masks_y = torch.isnan(y).int()
        masks_y = 1 - masks_y

        X = X.masked_fill(masks_X == 0, -2)
        y = y.masked_fill(masks_y == 0, -1e1) # This does not really matter as it is gonna be masked out in loss function

        return (X, masks_X, time), (y, masks_y)
