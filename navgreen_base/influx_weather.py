import pandas as pd
import os
import logging

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

weather_columns = ["TEMPERATURE", "HUMIDITY", "WIND_SPEED", "WIND_DIRECTION", "SKY", "predicted", "probabilities", "DATETIME"]

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
# Configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(stream_handler)

# Import organisation
organization = os.environ.get('Organization_influx')
# Bucket must be defined from the user using function 'set_bucket'
bucket = None


# Define bucket used by the following functions
def set_bucket(b):
    global bucket
    bucket = b
    return bucket


# Establish connection with InfluxDb
def establish_influxdb_connection():
    # Import credentials
    url = os.environ.get('Url_influx_db')
    auth_token = os.environ.get('Auth_token')

    return influxdb_client.InfluxDBClient(url=url, token=auth_token, org=organization)


def make_point(measurement, row, value_columns, tag_columns):
    p = influxdb_client.Point(measurement)
    #p.time(row["DATETIME"])
    p.time(row["FORECAST_DATETIME"])

    # Tag with the state of the valves, as context
    for col in tag_columns:
        p.tag(col, row[col])
    # Add the sensor data fields
    for col in value_columns:
        p.field(col, row[col])

    return p


def write_data(row, influx_client):
    """
    Writes one row to a specified bucket. The bucket that will
    be used should be set using `set_bucket` before the first
    invocation of this method. It does not need to be set again
    for subsequent invocations.
    """
    api = influx_client.write_api(write_options=SYNCHRONOUS)

    p = make_point("weather", row, weather_columns, [])
    api.write(bucket=bucket, org=organization, record=p)


def read_data(influx_client, start=0):
    """
    Reads data from a specified bucket and stores it in a DataFrame.
    The bucket that will be used should be set using `set_bucket`
    before the first invocation of this method. It does not need
    to be set again for subsequent invocations.
    :return: The DataFrame
    """
    # Supress warning about not having used pivot function
    # to optimize processing by pandas
    warnings.simplefilter("ignore", MissingPivotFunction)

    api = influx_client.query_api()
    query = f'from(bucket: "{bucket}") |> range(start: {start})'
    data = api.query_data_frame(org=organization, query=query)

    if isinstance(data, pd.DataFrame):
        df = data.pivot(index='_time', columns='_field', values='_value')
        # Reset the index to make the '_time' column a regular column
        df.reset_index(inplace=True)
        df.columns.name = None
        df = df.rename(columns={'_time': 'DATETIME'})

    elif isinstance(data, list):
        dfs = dict()  # []
        for datum in data:
            # Pivot the DataFrame to separate fields into different columns
            df = datum.pivot(index='FORECAST_DATETIME', columns='_field', values='_value')
            # Reset the index to make the 'FORECAST_DATETIME' column a regular column
            df.reset_index(inplace=True)
            df.columns.name = None

            # Each dataframe may have rows of different dates, so split them and organize them by date
            for index, row in df.iterrows():
                row_df = pd.DataFrame([row])
                row_df.reset_index(inplace=True)
                try:
                    dfs[row_df.FORECAST_DATETIME.iloc[0]].append(row_df)
                except KeyError:
                    dfs[row_df.FORECAST_DATETIME.iloc[0]] = [row_df]

        # Combine all the dataframes of the same datetime to one dataframe
        all_dfs = []
        for _, df_list in dfs.items():
            df = pd.concat(df_list, axis=1, join='outer', sort=False)
            df = df.rename(columns={'FORECAST_DATETIME': 'DATETIME'})
            # If conflicting values for a column, keep the non-NaN values
            df = df.groupby(df.columns, axis=1).apply(lambda x: x.ffill(axis=1).bfill(axis=1).iloc[:, 0])
            df = df.drop(columns=['index'])

            all_dfs.append(df)

        # Now concat all the dataframes as rows (different datetimes)
        df = pd.concat(all_dfs, axis=0, ignore_index=True)
        df = df.rename(columns={'FORECAST_DATETIME': 'DATETIME'})
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    return df


def delete_data(influx_client):
    """
    Deletes all data from a specified bucket.
    The bucket that will be used should be set using `set_bucket`
    before the first invocation of this method. It does not need
    to be set again for subsequent invocations.
    :return: None
    """
    api = influx_client.delete_api()
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="weather"')
