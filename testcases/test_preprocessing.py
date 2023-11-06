import unittest
import os

import numpy as np
import pandas as pd

from navgreen_base import write_data, delete_data, read_data
from live_data_analysis import temp_sensors, pressure, solar


class TestPreproc( unittest.TestCase ):


    @classmethod
    def setUpClass( cls ):
        # Import credentials
        cls._url = os.environ.get('Url_influx_db')
        cls._org = os.environ.get('Organization_influx')
        cls._auth_token = os.environ.get('Auth_token')
        # Ignore env variable, always use this bucket for tests
        cls._bucket = "test_bucket"
        # Load sample
        cls._test_data_path = "./testcases/sample_data/"
        cls._sample_input = pd.read_csv( cls._test_data_path+"sample_input.csv" )
    # end def setUpClass()

#    @classmethod
#    def tearDownClass(cls):
#        cls._connection.destroy()


    def test_filters( self ):
        # Wipe clean the test bucket
        print("Started deleting...")
        delete_data( TestPreproc._url, TestPreproc._auth_token, TestPreproc._org, TestPreproc._bucket )
        print("Ended deleting.")

        # Read each data sample and write it to the test bucket
        # Function 'navgreen_base.write_data' does the preprocessing etc
        print("Started writing...")
        for _, row in TestPreproc._sample_input.iterrows():
            write_data( row, TestPreproc._url, TestPreproc._auth_token, TestPreproc._org, TestPreproc._bucket )
        print("Ended writing.")

        # Get the 'sample output' by querying the test bucket
        sample_output = read_data( TestPreproc._url, TestPreproc._auth_token, TestPreproc._org, TestPreproc._bucket )
        sample_output.to_csv( TestPreproc._test_data_path+"sample_output.csv" )

        # Check that the two dataframes have the same number of rows
        self.assertEqual( sample_output.shape[0], TestPreproc._sample_input.shape[0] )

        # Get the columns of the sample input that have all their values equal to np.nan
        all_nan_columns_input = TestPreproc._sample_input.columns[TestPreproc._sample_input.isna().all()]

        # Delete the NaN columns from the DataFrame as well as the 'index', the 'Date_time_local'
        # because they do not exist in the sample_output
        cols_to_del = list(all_nan_columns_input) + ['index', 'Date_time_local']
        sample_input = TestPreproc._sample_input.drop(columns=cols_to_del)

        columns_unique_to_input = set(sample_input.columns) - set(sample_output.columns)
        columns_unique_to_output = set(sample_output.columns) - set(sample_input.columns)

        # sample output should not have different values
        self.assertEqual( columns_unique_to_output, set() )

        # If sample input still has different columns:
        # Check if their values are outliers & were eliminated by the pre-processing
        if columns_unique_to_input != set():
            for column in columns_unique_to_input:
                # Check that it is indeed in the columns to which pre-processing is applied
                self.assertTrue( (column in temp_sensors) or (column in pressure) )
                # Check that all their values are either np.nan or outliers
                if column in temp_sensors:
                    self.assertTrue(
                        ((pd.isna(sample_input[column])) |
                         (sample_input[column] < -20.0) |
                         (sample_input[column] > 100.0)).all() )
                else:
                    self.assertTrue(
                        ((pd.isna(sample_input[column])) |
                         (sample_input[column] < 0.0) |
                         (sample_input[column] > 35.0)).all() )

                # If it passed the check, it can be dropped from the sample input
                sample_input = sample_input.drop(columns=[column])

        # Check that the two dataframes have the same number of columns
        self.assertEqual( sample_output.shape[1], sample_input.shape[1] )

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

        # Compare the two DataFrames
        self.assertTrue( sample_input.compare(sample_output).empty )

    # end def test_filters()

# end class TestPreproc()
