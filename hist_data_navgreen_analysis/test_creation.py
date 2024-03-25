import unittest
import logging
import os

import numpy as np
import pandas as pd


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


class TestCreate( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):

        cls._concatenated_data = pd.DataFrame()
        cls._missing_columns = []
        cls._setpoints = ['T_CHECKPOINT_SPACE_HEATING_MODBUS', 'T_CHECKPOINT_DHW_MODBUS']
        cls._radiation = ['DIFFUSE_SOLAR_RADIATION']

        # Get a list of all CSV files in the folder
        csv_files = sorted([f.path for f in os.scandir("./hist_data_navgreen_analysis/data_hist_navgreen/") if f.is_file() and f.name.endswith('.csv')])

        # Iterate through each CSV file in the folder
        for i, file in enumerate(csv_files):
            try:
                df = pd.read_csv(file)

                # Check for missing columns and add them with np.nan padding
                # There are going to be missing columns as we added measurements periodically
                diff_columns = set(df.columns) - set(cls._concatenated_data.columns)
                for col in diff_columns:
                    cls._concatenated_data[col] = np.nan

                if i != 0: # In first iteration everything is going to be different
                    cls._missing_columns += diff_columns

                # Concatenate the DataFrame to the main DataFrame
                cls._concatenated_data = pd.concat([cls._concatenated_data, df], ignore_index=True)

            except pd.errors.ParserError as e:
                logger.error(f"Error reading {file}: {e}")

        # We only store the checkpoints when they have a new value otherwise it is NaN
        # So here we fill the column with its appropriate value
        for col in cls._setpoints:
            cls._concatenated_data[col] = cls._concatenated_data[col].ffill()

        # Save the final concatenated DataFrame to a new CSV file
        cls._concatenated_data.to_csv('./hist_data_navgreen_analysis/concatenated_data.csv', index=False)

    # end def setUpClass()

    def test_missing_columns( self ):
        # Check that the columns that differ within the different dataframes are the actual ones we added
        self.assertEqual(sorted(TestCreate._missing_columns), sorted(TestCreate._setpoints+TestCreate._radiation))
    # end def test_missing_columns()

    def test_checkpoint_fill( self ):
        # Check that we filled the numbers in the dataframe correctly
        for col in TestCreate._setpoints:
            first_non_nan_index = TestCreate._concatenated_data.loc[TestCreate._concatenated_data[col].notna()].index[0]

            # Assert that every row after the first non-nan row has no nan values in the specified column
            self.assertTrue(all(TestCreate._concatenated_data.loc[first_non_nan_index + 1:, col].notna()), (
                "Column {} has nan values in rows after the first non-nan row.".format(col)))
    # end def test_checkpoint_fill()

    def test_change_date( self ):
        # Check that the days we first encounter the columns that were added later are the correct ones
        for col in TestCreate._setpoints:
            first_row_with_value = TestCreate._concatenated_data.loc[TestCreate._concatenated_data[col].notna()].iloc[0]
            self.assertEqual(first_row_with_value['DATETIME'][:10], "2023-12-15")

        first_row_with_value = \
        TestCreate._concatenated_data.loc[TestCreate._concatenated_data['DIFFUSE_SOLAR_RADIATION'].notna()].iloc[0]
        self.assertEqual(first_row_with_value['DATETIME'][:10], "2024-02-16")
    # end def test_change_date()

# end class TestCreate()
