import pandas as pd
import numpy as np
from datetime import datetime
import os


def read_csv(weather_path):
    """
    The weather dataframe consists of the current month and year.
    Note that the last sampled day of each month the .csv is going to consist of the hours 00:00 and 03:00 of the
    next day (1st of the next month).
    This modeling is intended for this processing, but for monthly (or other) processing an aggregation
    of the .csvs should be done.
    """

    current_date = datetime.now()
    csv_filename = os.path.join(weather_path, current_date.strftime('%B_%Y_2_30.csv').lower())
    df = pd.read_csv(csv_filename)

    # Get the last 8 rows (8 x 3 hrs)
    df_predictions_today = df.tail(8).reset_index()

    df_predictions_today['ACCESS_DATETIME'] = pd.to_datetime(df_predictions_today['ACCESS_DATETIME'],
                                                             format='%d/%m/%Y %H:%M:%S')

    # Check if the values were actually accessed the same day as this script is run
    if not all(df_predictions_today['ACCESS_DATETIME'].dt.date == current_date.date()):
        raise ValueError("Today's weather predictions do not exist.")

    return df_predictions_today


def filter_weather(df, rename_columns, create_columns, columns):

    df_filtered = df.rename(columns=rename_columns)

    for col in create_columns:
        df_filtered[col] = np.nan

    df_filtered = df_filtered[columns]

    return df_filtered


if __name__ == '__main__':

    weather_csv = read_csv(weather_path='../')
    # THE COLUMNS ARE DEFINITELY WRONG, WE ARE JUST CONSTRUCTING THE ARCHITECTURE OF THE PIPELINE
    rename_columns = {'FORECAST_DATETIME': 'DATETIME', 'TEMPERATURE': 'temp', 'HUMIDITY': 'humidity',
                      'WIND_SPEED': 'wind_speed'}
    create_columns = ['pressure', 'feels_like', 'rain_1h']
    keep_columns = [rename_columns[key] for key in rename_columns] + create_columns

    weather_csv = filter_weather(df=weather_csv, columns=keep_columns,
                                 create_columns=create_columns, rename_columns=rename_columns)

    print(weather_csv)