from navgreen_base import process_data

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


def load_df(df_path, hp_cols, pvt_cols, parse_dates, hist_data, normalize=True, grp=None, stats=None):
    """
    Loads the data from the historical data DataFrame
    :return: whole dataframe, dataframe for hp, dataframe for solar
    """
    df = pd.read_csv(df_path, parse_dates=parse_dates, low_memory=False)
    df = process_data(df, hist_data=hist_data)

    if not hist_data:
        logger.info(f'DHW_BOTTOM min: {df["DHW_BOTTOM"].min()}, DHW_BOTTOM max: {df["DHW_BOTTOM"].max()}')

    logger.info("All data: {} rows".format(len(df)))

    if hist_data:
        df = df[(df['DATETIME'] > '2022-08-31') & (df['DATETIME'] < '2023-09-01')]
        logger.info("12 months data: {} rows".format(len(df)))

    df = df.loc[df["FOUR_WAY_VALVE"] == "HEATING"] if hist_data else df.loc[df["FOUR_WAY_VALVE"] == "1.0"]
    logger.info("HEATING: {} rows".format(len(df)))

    # Add calculated columns (thermal flows)
    df["Q_CON_HEAT"] = 4.18 * (998.0 / 3600.0 * df["FLOW_CONDENSER"]) * (df["WATER_OUT_COND"] - df["WATER_IN_COND"])
    df["Q_PVT"] = ((3.6014 + 0.004 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.000002 *
                    pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) *
                   ((1049.0 - 0.475 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.0018 *
                     pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) / 3600.0 * df["FLOW_PVT"]) *
                   (df["PVT_OUT"] - df["PVT_IN"]))

    # Drop unneeded columns
    for c in df.columns:
        if (c not in hp_cols) and (c not in pvt_cols):
            df.drop(c, axis="columns", inplace=True)

    mean_stds = {} if stats is None else stats

    # Normalize each column
    if normalize:
        for c in [c for c in df.columns if c != "DATETIME"]:
            series = df[c]
            logger.info(f'column {c} -> {round(df[c].mean(), 1)}, {round(df[c].std(), 1)}')
            if stats is None: mean_stds[c] = (series.mean(), series.std())
            df[c] = (series - mean_stds[c][0]) / mean_stds[c][1]
            # assert round(df[c].mean(), 1) == 0.0 and round(df[c].std(), 1) == 1.0

    # If group data by time
    df = df.set_index("DATETIME").resample(grp).mean().sort_index() if grp else df.set_index("DATETIME").sort_index()
    logger.info("Grouped: {} rows".format(len(df)))

    # Represent unwanted data as irregularly sampled
    df_hp = df[[c for c in hp_cols if c != "DATETIME"]]
    df_hp.loc[df_hp["POWER_HP"] > 1.0, :] = float('nan')
    # df_hp = df_hp[df_hp["POWER_HP"] > 1.0]
    logger.info("HP, POWER > 1: {} rows".format(len(df_hp)))

    df_pvt = df[[c for c in pvt_cols if c != "DATETIME"]]
    df_pvt.loc[df_pvt["PYRANOMETER"] > 0.15, :] = float('nan')
    # df_pvt = df_pvt[df_pvt["PYRANOMETER"] > 0.15]
    logger.info("PV, PYRAN > 0.15: {} rows".format(len(df_pvt)))

    if stats is not None: mean_stds = None
    return (df, df_hp, df_pvt), mean_stds


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length, X_cols, y_cols):
        self.sequence_length = sequence_length

        df = dataframe
        df['Datetime'] = df.index.astype('int64')
        # Normalize Unix time between 0 and 1
        min_time, max_time = df['Datetime'].min(), df['Datetime'].max()
        epsilon = 1e12 # Small epsilon value to prevent reaching exactly 1
        df['Datetime'] = (df['Datetime'] - min_time) / (max_time - min_time + epsilon)

        # Check if any column in y_cols contains NaN values for each row
        has_nan_in_y_cols = df[y_cols].isna().any(axis=1)
        # Replace rows with NaN values in any of the y_cols with NaN values
        df.loc[has_nan_in_y_cols, :] = float('nan')

        self.X, self.y = df[X_cols], df[y_cols]
        self.time = df['Datetime']

    def __len__(self):
        return self.X.shape[0] - self.sequence_length + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length
        X = self.X.iloc[start_idx:end_idx].values
        y = self.y.iloc[start_idx:end_idx].values
        time = self.time.iloc[start_idx:end_idx].values
        time = np.nan_to_num(time, nan=0)

        masks_X = np.isnan(X).astype(int)
        masks_X = 1 - masks_X

        masks_y = np.isnan(y).astype(int)
        masks_y = 1 - masks_y

        # Convert the sequence to a PyTorch tensor
        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        masks_X, masks_y = torch.tensor(masks_X), torch.tensor(masks_y)
        time = torch.FloatTensor(time)

        X = X.masked_fill(masks_X == 0, 0)
        y = y.masked_fill(masks_y == 0, 0)

        return (X, masks_X, time), y
