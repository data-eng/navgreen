import os
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def create_weather_dataframe():

    verbose_analysis = False

    folder_path = '../data/data'

    dataframes = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    df['ACCESS_DATETIME'] = pd.to_datetime(df['ACCESS_DATETIME'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

    if verbose_analysis:
        # Find what missing dates we have in our data
        df['ACCESS_DATE'] = df['ACCESS_DATETIME'].dt.date
        valid_dates = df['ACCESS_DATE'].dropna().unique()
        start_date, end_date = valid_dates.min(), valid_dates.max()
        full_date_range = pd.date_range(start=start_date, end=end_date).date
        missing_dates = set(full_date_range) - set(valid_dates)
        print(f"Missing Dates: {missing_dates}")

    if verbose_analysis:
        # Calculate the percentage of NaN values in each column
        print('NaN percentage (%) of each column:')
        nan_percentage = df.isna().mean() * 100
        print(nan_percentage)

    # 'ACCESS_DATETIME'   : full date
    # 'FORECAST_DATETIME' : full date
    # 'SUNRISE'           : time
    # 'SUNSET'            : time
    # 'TEMPERATURE'       : integer, numerical
    # 'HUMIDITY'          : integer, numerical
    # 'WIND_SPEED'        : integer, numerical but in 1-to-1 correspondence with beaufort
    # 'BEAUFORT'          : integer, numerical, scale
    # 'WIND_DIRECTION'    : integer, categorical (E, W, EW etc.)
    # 'SKY'               : integer, categorical

    if verbose_analysis:
        for column in df.columns:
            print(f"Unique values in column '{column}':")
            print(sorted(df[column].unique()))
            print()

    # Drop 'BEAUFORT' column since we have wind speed as an alternative
    df = df.drop('BEAUFORT', axis=1)

    # Sort dataframe to ensure everything is in the correct order
    df['FORECAST_DATETIME'] = pd.to_datetime(df['FORECAST_DATETIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.sort_values(by='FORECAST_DATETIME')

    df['DATETIME'] = df['FORECAST_DATETIME']
    df = df.drop(['FORECAST_DATETIME', 'ACCESS_DATETIME'], axis=1)

    if verbose_analysis:
        print(df.columns)

    file_path = os.path.join('../data', 'data_meteo.csv')
    df.to_csv(file_path, index=False)

    return df


def process_weather_df(df, drop_columns):
    df = df.drop(drop_columns, axis=1)

    return df
