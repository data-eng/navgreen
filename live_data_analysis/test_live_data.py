import influxdb_client
import pandas as pd
import numpy as np
import os

from data_and_connection import write_data
from data_and_connection import temp_sensors, pressure, solar

import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)


def delete_data(url, token, organization, bucket):
    """
    Deletes all data from a specified bucket.
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: None
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.delete_api()
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="temperature"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="pressure"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="flow"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="solar"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="other"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="power"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="control"')


def read_data(url, token, organization, bucket):
    """
    Reads data from a specified bucket and stores it in a DataFrame
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: The DataFrame
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.query_api()
    query = f'from(bucket: "{bucket}") |> range(start: 0)'
    print("Started reading...")

    data = api.query_data_frame(org=organization, query=query)

    dfs = []
    for datum in data:
        # Pivot the DataFrame to separate fields into different columns
        df = datum.pivot(index='_time', columns='_field', values='_value')
        # Reset the index to make the '_time' column a regular column
        df.reset_index(inplace=True)
        df.columns.name = None

        dfs += [df]

    df1 = dfs[0]
    df2 = dfs[1]
    df = pd.concat([df1, df2], axis=1, join='outer', sort=False)
    df = df.rename(columns={'_time': 'DATETIME'})
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    print("Ended reading.")

    return df


if __name__ == "__main__":

    # Load 'sample input'
    sample_input = pd.read_csv("./test/sample_input.csv")

    # Import credentials
    url = os.environ.get('Url_influx_db')
    org = os.environ.get('Organization_influx')
    auth_token = os.environ.get('Auth_token')
    bucket = os.environ.get('Bucket')

    assert bucket == 'test_bucket'

    # Wipe clean the test bucket
    print("Started deleting...")
    delete_data(url, auth_token, org, bucket)
    print("Ended deleting.")

    # Read each data sample and write it to the test bucket
    # Function 'write_data' is imported from the ingestion script and does the preprocessing etc
    print("Started writing...")
    for _, row in sample_input.iterrows():
        write_data(row, url, auth_token, org, bucket)
    print("Ended writing.")
    
    # Get the 'sample output' by querying the test bucket
    sample_output = read_data(url, auth_token, org, bucket)
    sample_output.to_csv('./test/sample_output.csv')

    # Check that the two dataframes have the same number of rows
    assert sample_output.shape[0] == sample_input.shape[0]

    # Get the columns of the sample input that have all their values equal to np.nan
    all_nan_columns_input = sample_input.columns[sample_input.isna().all()]

    # Delete the NaN columns from the DataFrame as well as the 'index', the 'Date_time_local'
    # because they do not exist in the sample_output
    cols_to_del = list(all_nan_columns_input) + ['index', 'Date_time_local']
    sample_input = sample_input.drop(columns=cols_to_del)

    columns_unique_to_input = set(sample_input.columns) - set(sample_output.columns)
    columns_unique_to_output = set(sample_output.columns) - set(sample_input.columns)

    # sample output should not have different values
    assert columns_unique_to_output == set()

    # If sample input still has different columns:
    # Check if their values are outliers & were eliminated by the pre-processing
    if columns_unique_to_input != set():
        for column in columns_unique_to_input:
            # Check that it is indeed in the columns to which pre-processing is applied
            assert ((column in temp_sensors) or (column in pressure))
            # Check that all their values are either np.nan or outliers
            if column in temp_sensors:
                assert ((pd.isna(sample_input[column])) | (sample_input[column] < -20.0) |
                        (sample_input[column] > 100.0)).all()
            else:
                assert ((pd.isna(sample_input[column])) | (sample_input[column] < 0.0) |
                        (sample_input[column] > 35.0)).all()

            # If it passed the check, it can be dropped from the sample input
            sample_input = sample_input.drop(columns=[column])

    # Check that the two dataframes have the same number of columns
    assert sample_output.shape[1] == sample_input.shape[1]

    sample_output = sample_output.reset_index()
    sample_input = sample_input.reset_index()

    # Make sure the columns are ordered in the same way
    sample_input = sample_input[sample_output.columns]

    # Also, convert date of sample input to the influx format
    sample_input['DATETIME'] = pd.to_datetime(sample_input['DATETIME']).dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

    for col in sample_input.columns:
        # Treat the 'solar' outliers
        if col in solar:
            sample_input.loc[sample_input[col] > 2.0, col] = 0.0
        # Treat the 'temperature' outliers
        if col in temp_sensors:
            sample_input.loc[(sample_input[col] < -20.0) | (sample_input[col] > 100.0), col] = np.nan
        # Treat the 'pressure' outliers
        if col in pressure:
            sample_input.loc[(sample_input[col] < 0.0) | (sample_input[col] > 35.0), col] = np.nan

    comparison = sample_input.compare(sample_output)

    # Compare the two DataFrames
    assert comparison.empty
