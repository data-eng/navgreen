import os
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import logging

from public_weather_installation.test import main_loop
from navgreen_base import establish_influxdb_connection, set_bucket, write_data

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s')
# Set log file, its level and format
file_handler = logging.FileHandler('./logger.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
# Set stream its level and format
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


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

    df_predictions_today['FORECAST_DATETIME'] = pd.to_datetime(df_predictions_today['FORECAST_DATETIME'])

    # Check if the values were actually accessed the same day as this script is run
    #if not all(df_predictions_today['ACCESS_DATETIME'].dt.date == current_date.date()):
    #    raise ValueError("Today's weather predictions do not exist.")

    return df_predictions_today


def filter_weather(df, rename_columns, create_columns, columns):

    df_filtered = df.rename(columns=rename_columns)

    for col in create_columns:
        df_filtered[col] = np.nan

    df_filtered = df_filtered[columns]

    return df_filtered


def model_predictions(input_data, model_pth):
    return main_loop(df=input_data, model_pth=model_pth)


def write_to_influxdb(qpvt_predictions, forecast_datetime):
    # Establish connection with InfluxDb
    influx_client = establish_influxdb_connection()
    # Set the bucket
    _ = set_bucket(os.environ.get('Bucket_Model'))

    reconnect_interval = 30

    qpvt_predictions = pd.DataFrame(qpvt_predictions)
    # Convert DATETIME to a format suitable for InfluxDB
    # qpvt_predictions['DATETIME'] = pd.to_datetime(qpvt_predictions['DATETIME']) - timedelta(days=1)
    qpvt_predictions['DATETIME'] = qpvt_predictions['DATETIME'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    qpvt_predictions['FORECAST_DATETIME'] = forecast_datetime.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    # Same with probabilities
    print(qpvt_predictions)
    qpvt_predictions['probabilities'] = qpvt_predictions['probabilities'].apply(json.dumps)

    while True:
        try:
            # Try and write data to InfluxDB
            for idx in range(qpvt_predictions.shape[0]):
                write_data(qpvt_predictions.iloc[idx], influx_client)

            break

        except Exception as e:
            logger.error(f"Failed to write to InfluxDB: {str(e)}")
            # If write was not successful, try again later
            logger.info(f"Sleeping for {reconnect_interval} seconds..")
            time.sleep(reconnect_interval)
            continue

    logger.info("Written to InfluxDB!")


if __name__ == '__main__':

    weather_csv = read_csv(weather_path='./weather_predictions/data/')
    # THE COLUMNS ARE DEFINITELY WRONG, WE ARE JUST CONSTRUCTING THE ARCHITECTURE OF THE PIPELINE
    rename_columns = {'FORECAST_DATETIME': 'FORECAST_DATETIME', 'ACCESS_DATETIME' : 'DATETIME', 'TEMPERATURE': 'temp', 'HUMIDITY': 'humidity',
                      'WIND_SPEED': 'wind_speed'}
    create_columns = ['pressure', 'feels_like', 'rain_1h']
    keep_columns = [rename_columns[key] for key in rename_columns] + create_columns

    # Get the weather DataFrame
    weather_csv = filter_weather(df=weather_csv, columns=keep_columns,
                                 create_columns=create_columns, rename_columns=rename_columns)

    weather_csv['DATETIME'] = pd.to_datetime(weather_csv['DATETIME'])
    weather_csv['FORECAST_DATETIME'] = pd.to_datetime(weather_csv['FORECAST_DATETIME'])
    # print(weather_csv['FORECAST_DATETIME'])

    # Feed forward the weather to obtain the 3hr-Q_PVT predictions
    predictions = model_predictions(weather_csv, 'weather_predictions/communication_PLC/public_weather_installation/transformer.pth')

    # Write the predictions to InfluxDB
    write_to_influxdb(qpvt_predictions=predictions, forecast_datetime=weather_csv['FORECAST_DATETIME'])
