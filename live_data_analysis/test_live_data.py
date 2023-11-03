import pandas as pd
import numpy as np
import os

from navgreen_base import write_data, delete_data, read_data
from . import temp_sensors, pressure, solar
from . import TESTDATAPATH


def tester():

    # Load 'sample input'
    sample_input = pd.read_csv(TESTDATAPATH+"sample_input.csv")

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
    sample_output.to_csv( TESTDATAPATH+"sample_output.csv" )

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
