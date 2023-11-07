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
        url = os.environ.get('Url_influx_db')
        auth_token = os.environ.get('Auth_token')
        org = os.environ.get('Organization_influx')
        # Ignore env variable, always use this bucket for tests
        bucket = "test_bucket"

        # Load sample
        cls.test_data_path = "./testcases/sample_data/"
        cls._sample_input = pd.read_csv( cls.test_data_path+"sample_input.csv" )

        # Wipe clean the test bucket
        print( "Emptying test_bucket... ", end="" )
        delete_data( url, auth_token, org, bucket )
        print("DONE.")

        # Read each data sample and write it to the test bucket
        # Function 'navgreen_base.write_data' does the preprocessing etc
        print( "Writing test samples to test_bucket... ", end="" )
        for _, row in cls._sample_input.iterrows():
            write_data( row, url, auth_token, org, bucket )
        print("DONE.")

        # Get the 'sample output' by querying the test bucket
        print( "Reading test samples back from test_bucket... ", end="" )
        cls._sample_output = read_data( url, auth_token, org, bucket )
        print("DONE.")

        # Get the columns of the sample input that have all their values equal to np.nan
        all_nan_columns_input = cls._sample_input.columns[cls._sample_input.isna().all()]

        # Delete the NaN columns from the DataFrame as well as the 'index', the 'Date_time_local'
        # because they do not exist in the sample_output
        cols_to_del = list(all_nan_columns_input) + ['index', 'Date_time_local']
        cls._sample_input.drop( columns=cols_to_del, inplace=True )

    # end def setUpClass()

    def test_sample_output( self ):
        # Check that nothing changes in the sample_output
        sample_output_stored = pd.read_csv( TestPreproc.test_data_path+"sample_output.csv", index_col=0 )
        self.assertTrue(sample_output_stored.compare(TestPreproc._sample_output).empty)
    # end def test_sample_output()

    def test_rows( self ):
        # Check that the two dataframes have the same number of rows
        self.assertEqual( TestPreproc._sample_output.shape[0], TestPreproc._sample_input.shape[0] )
    # end def test_rows()

    def test_extra_columns( self ):
        sample_input = TestPreproc._sample_input
        sample_output = TestPreproc._sample_output
        columns_unique_to_output = set(sample_output.columns) - set(sample_input.columns)
        self.assertEqual( columns_unique_to_output, set() )
    # end def test_extra_columns()

    def test_missing_columns( self ):
        sample_input = TestPreproc._sample_input
        sample_output = TestPreproc._sample_output
        columns_unique_to_input = set(sample_input.columns) - set(sample_output.columns)
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
    # end def test_missing_columns()

    def test_columns_order( self ):
        sample_output = TestPreproc._sample_output.reset_index()
        sample_input = TestPreproc._sample_input.reset_index()
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
    # end def test_columns_order()

# end class TestPreproc()
