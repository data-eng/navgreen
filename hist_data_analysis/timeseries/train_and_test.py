import pandas as pd
from torch.utils.data import Dataset, DataLoader

from .data_loader import load_df, TimeSeriesDataset

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



hp_cols = ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"]
pvt_cols = ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"]



# POWER_HP= f (BTES_TANK, DHW_BUFFER, WATER_IN_EVAP, WATER_IN_COND, OUTDOOR_TEMP, COMPRESSOR_HZ)
# Q_CON_HEAT=( WATER_OUT_COND, WATER_IN_COND, FLOW_CONDENSER, DHW_BUFFER, POWER_HP)

# POWER_PVT= f (OUTDOOR_TEMP, PYRANOMETER, PVT_IN, PVT_OUT)
# Q_PVT= f(OUTDOOR_TEMP, PYRANOMETER, PVT_IN, PVT_OUT, FLOW_PVT)


def main_loop():

    df_path = "data/DATA_FROM_PLC.csv"
    (df, df_hp, df_pvt), _ = load_df(df_path, hp_cols, pvt_cols, normalize=True, grp="2T")  # 2T is 2 minutes

    df_hp.to_csv("hp_df.csv")

    print(df.shape, df.columns)
    print(df_hp.shape, df_hp.columns)
    print(df_pvt.shape, df_pvt.columns)

    df = pd.read_csv("hp_df.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    # Example usage:
    # Assuming 'df' is your sorted DataFrame with timestamp as index

    # Specify the sequence length
    sequence_length = 5

    X_hp_cols = ["BTES_TANK", "DHW_BUFFER"]
    y_hp_cols = ["POWER_HP", "Q_CON_HEAT"]

    # X_pvt_cols = ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"]
    # y_pvt_cols = ["POWER_PVT", "Q_PVT"]

    # Create an instance of the TimeSeriesDataset
    dataset = TimeSeriesDataset(dataframe=df, sequence_length=sequence_length, X_cols=X_hp_cols, Y_cols=y_hp_cols)

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Iterate through the dataloader
    print(f'Dataset size is {len(dataloader)}')
    for X, y in dataloader:
        print("Sequence:")
        print(X)
        print(y)

        break
    # X, y1, y2 = prepare(train)