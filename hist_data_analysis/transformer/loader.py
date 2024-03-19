import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import logging
from navgreen_base import process_data
from hist_data_analysis import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

data = {
    "hp": {
        "cols": ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"],
        "X": ["BTES_TANK", "DHW_BUFFER"],
        "y": ["POWER_HP", "Q_CON_HEAT"],
        "ignore": [("DATETIME", None), ("POWER_HP", 1.0)]
    },
    "pv": {
        "cols": ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"],
        "X": ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"],
        "y": ["POWER_PVT", "Q_PVT"],
        "ignore": [("DATETIME", None), ("PYRANOMETER", 0.15)]
    }
}

def load(path, parse_dates, normalize=True, grp=None, agg=None, hist_data=True):
    """
    Loads and preprocesses data from a CSV file.
    :param path: path to the CSV file
    :param parse_dates: columns to parse as dates in the dataframe
    :param normalize: normalization flag
    :param grp: frequency to group data by
    :param agg: aggregation function
    :param hist_data: hist_data flag
    :return: dataframe
    """
    df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    df = process_data(df, hist_data=hist_data)

    logger.info("All data: {} rows".format(len(df)))

    if not hist_data:
        logger.info(f'DHW_BOTTOM min: {df["DHW_BOTTOM"].min()}, DHW_BOTTOM max: {df["DHW_BOTTOM"].max()}')
        df.loc[df["FOUR_WAY_VALVE"] == "1.0"]

    if hist_data:
        df = df[(df['DATETIME'] > '2022-08-31') & (df['DATETIME'] < '2023-09-01')]
        logger.info("12 months data: {} rows".format(len(df)))
        df = df.loc[df["FOUR_WAY_VALVE"] == "HEATING"]
        
    logger.info("HEATING: {} rows".format(len(df)))

    df["Q_CON_HEAT"] = 4.18 * (998.0 / 3600.0 * df["FLOW_CONDENSER"]) * (df["WATER_OUT_COND"] - df["WATER_IN_COND"])
    df["Q_PVT"] = ((3.6014 + 0.004 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.000002 *
                    pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) *
                   ((1049.0 - 0.475 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.0018 *
                     pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) / 3600.0 * df["FLOW_PVT"]) *
                   (df["PVT_OUT"] - df["PVT_IN"]))
    
    for c in df.columns:
        if (c not in data["hp"]["cols"]) and (c not in data["pv"]["cols"]):
            df.drop(c, axis="columns", inplace=True)
    df.dropna(inplace=True)
    logger.info("No NaNs: {} rows".format(len(df)))

    df = utils.normalize(df) if normalize else df
    df = utils.aggregate(df, grp, func=agg)

    return df

def prepare(df, system):
    """
    Prepares the dataframe for training by filtering columns and saving to CSV.
    :param df: dataframe
    :param system: str name of the system (HP or PV)
    :return: dataframe
    """
    params = data[system]
    name = "df_" + system + ".csv"

    for column, threshold in params["ignore"]:
        df = utils.filter(df, column=column, threshold=threshold) 

    df.to_csv(name)
    df = pd.read_csv(name, parse_dates=['DATETIME'], index_col='DATETIME')

    return df

def split(dataset, vperc=0.2):
    """
    Splits a dataset into training and validation sets.
    :param dataset: dataset
    :param vperc: percentage of data to allocate for validation
    :return: tuple containing training and validation datasets
    """
    ds_size = len(dataset)

    valid_size = int(vperc * ds_size)
    train_size = ds_size - valid_size

    return random_split(dataset, [train_size, valid_size])
    
class TSDataset(Dataset):
    def __init__(self, dataframe, seq, X, y):
        """
        Initializes a time series dataset.
        :param dataframe: dataframe
        :param seq: length of the input sequence
        :param X: input features names
        :param y: target variables names
        """
        self.seq = seq
        self.X = dataframe[X]
        self.y = dataframe[y]

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.X.shape[0] - self.seq + 1

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.
        :param idx: index of the sample
        :return: tuple containing input features sequence and target variables
        """
        start_idx = idx
        end_idx = idx + self.seq
        X = self.X.iloc[start_idx:end_idx].values
        y = self.y.iloc[start_idx:end_idx].values
        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        return X, y