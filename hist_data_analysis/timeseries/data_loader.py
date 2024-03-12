from navgreen_base import process_data

import pandas as pd
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

    # Drop unneeded columns and then drop rows with NaN
    # to avoid dropping rows with NaNs in cols we don't need anyway
    # Cannot be done on df_hp,df_pv views, modifying views is unsafe.
    for c in df.columns:
        if (c not in hp_cols) and (c not in pvt_cols):
            df.drop(c, axis="columns", inplace=True)
    df.dropna(inplace=True)
    logger.info("No NaNs: {} rows".format(len(df)))

    mean_stds = {} if stats is None else stats

    # Normalize each column
    if normalize:
        for c in [c for c in df.columns if c != "DATETIME"]:
            series = df[c]
            logger.info(f'column {c} -> {round(df[c].mean(), 1)}, {round(df[c].std(), 1)}')
            if stats is None: mean_stds[c] = (series.mean(), series.std())
            df[c] = (series - mean_stds[c][0]) / mean_stds[c][1]
            # assert round(df[c].mean(), 1) == 0.0 and round(df[c].std(), 1) == 1.0

    col = "DATETIME"
    print(f"Data type of DATETIME: {df[col].dtype}")
    # If group data by time
    df = df.set_index("DATETIME").resample(grp).mean().dropna().sort_index() if grp else df.set_index("DATETIME").sort_index()
    logger.info("Grouped: {} rows".format(len(df)))

    df_hp = df[[c for c in hp_cols if c != "DATETIME"]]
    df_hp = df_hp[df_hp["POWER_HP"] > 1.0]
    logger.info("HP, POWER > 1: {} rows".format(len(df_hp)))

    df_pvt = df[[c for c in pvt_cols if c != "DATETIME"]]
    df_pvt = df_pvt[df_pvt["PYRANOMETER"] > 0.15]
    logger.info("PV, PYRAN > 0.15: {} rows".format(len(df_pvt)))

    if stats is not None: mean_stds = None
    return (df, df_hp, df_pvt), mean_stds


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length, X_cols, Y_cols):
        self.sequence_length = sequence_length
        self.X = dataframe[X_cols]
        self.y = dataframe[Y_cols]

    def __len__(self):
        return self.X.shape[0] - self.sequence_length + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length
        X = self.X.iloc[start_idx:end_idx].values
        y = self.y.iloc[start_idx:end_idx].values

        # Convert the sequence to a PyTorch tensor
        X, y = torch.FloatTensor(X), torch.FloatTensor(y)

        return X, y
