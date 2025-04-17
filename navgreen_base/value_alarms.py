import logging

import pandas
import numpy

import navgreen_base

# Configure logger and set its level
logger = logging.getLogger( "value_alarms" )
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
# Configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(stream_handler)


def load_data(df_path):
    """
    Loads the data from the historical data DataFrame
    :return: whole dataframe, dataframe for hp, dataframe for solar
    """
    df = pandas.read_csv(df_path, parse_dates=["DATETIME"], low_memory=False)
    df = navgreen_base.process_data(df)

    return df
# end def load_data

def month_stats( df, incr_avg ):
    day00 = df.DATETIME.dt.day.iloc[0]
    day30 = df.DATETIME.dt.day.iloc[-1]
    print("{} {}".format(day00,day30))
    retv = []

    for col in navgreen_base.numerical_columns:
        mavg = df[col].mean()
        retv.append(mavg)
        if incr_avg is not None:
            assert len(incr_avg) == 1
            reldiffavg = mavg/incr_avg[col][0] - 1
            if reldiffavg < 0: reldiffavg = -reldiffavg
            if reldiffavg < 1: reldiffavg = 1/reldiffavg
            if reldiffavg > 42.0:
                print("{} is having a bad month, rel diff {}".format(col, reldiffavg))
        for day in range(day00, day30 + 1):
            df_daily = df[df.DATETIME.dt.day == day]
            reldiffavg = df_daily[col].mean()/mavg - 1
            if reldiffavg < 0: reldiffavg = -reldiffavg
            # Do not check for too small as with months, it happens
            if reldiffavg > 42.0:
                print( "{} is having a bad day, rel diff {}".format(col,reldiffavg) )
    dfretv = pandas.DataFrame([retv], columns=navgreen_base.numerical_columns)
    return dfretv
#end def month_stats

def different_dtypes_columns(df_old, df_new):
    # Get the columns with different data types
    different_dtypes = [(col, df_old[col].dtype, df_new[col].dtype) for col in df_old.columns if
                        df_old[col].dtype != df_new[col].dtype]

    # Print the results
    if different_dtypes:
        print("Columns with different data types:")
        for col, dtype1, dtype2 in different_dtypes:
            print(f"Column '{col}': {dtype1} (OLD DF) != {dtype2} (NEW DF)")
    else:
        print("No columns with different data types found.")
#end def different_dtypes_columns


def compare_string_columns(df_old, df_new):
    # Get columns that are of string type
    string_cols = [col for col in df_old.columns if df_old[col].dtype == 'object']

    if not string_cols:
        print("No string columns found.")
        return

    # Check for each string column if values are the same
    print()
    for col in string_cols:
        unique_values_df_old, unique_values_df_new = set(df_old[col].unique()), set(df_new[col].unique())

        if unique_values_df_old == unique_values_df_new:
            print(f"Values in column '{col}' are the same in both dataframes.")
        else:
            print(f"Column '{col}' has values {unique_values_df_old} in OLD DF and {unique_values_df_new} in NEW DF.")
#end compare_string_columns

def filter_months(old_df, new_df, months):

    new_df = new_df[new_df['DATETIME'].dt.month.isin(months)]
    old_df = old_df[old_df['DATETIME'].dt.month.isin(months)]

    return old_df, new_df
#end filter_months

def check_stats_within_magnitude(old_df, new_df, magnitude=10):

    print()
    for col in old_df.columns:
        # Skip non-numeric columns and DATETIME
        if not pandas.api.types.is_numeric_dtype(old_df[col]) or col == "DATETIME":
            continue


        if new_df[col].notna().any() and old_df[col].notna().any():
            # Calculate mean and standard deviation for each column
            mean_diff = abs(old_df[col].mean(skipna=True) - new_df[col].mean(skipna=True))
            std_diff = abs(old_df[col].std(skipna=True) - new_df[col].std(skipna=True))

            # Check if differences are within magnitude
            if mean_diff >= magnitude or std_diff >= magnitude:
                print(f"Mean({mean_diff:.4f}) or std ({std_diff:.4f}) difference in column '{col}' exceeds {magnitude}.")
            else:
                print(f"Mean({mean_diff:.4f}) or std ({std_diff:.4f}) difference in column '{col}' < {magnitude}.")
        else:
            print(f"Column '{col}' has no values within these months in neither dataframe.")
#end check_stats_within_magnitude

def main():
    print("\nCheck units within the historical data.\n")
    df = load_data(df_path="data/DATA_FROM_PLC_CONV.csv")
    dfavg = None
    for m in range(9, 21):
        df_monthly = df[df.DATETIME.dt.month == ((m - 1) % 12) + 1]
        logger.info("{} to {}".format(df_monthly.DATETIME.dt.date.iloc[0],
                                      df_monthly.DATETIME.dt.date.iloc[-1]))
        newavg = month_stats(df_monthly,dfavg)
        if dfavg is None: dfavg = newavg
        else: dfavg = (dfavg + newavg)/2

    print("\nCheck units and types between the new and old historical data.\n")
    old_df = load_data(df_path="data/DATA_FROM_PLC_CONV.csv")
    new_df = load_data(df_path="data/concatenated_data.csv")

    # Drop newly added columns
    new_df.drop(columns=['T_CHECKPOINT_DHW_MODBUS', 'T_CHECKPOINT_SPACE_HEATING_MODBUS', 'DIFFUSE_SOLAR_RADIATION'], inplace=True)
    assert sorted(old_df.columns) == sorted(new_df.columns)

    # Print columns with different values
    different_dtypes_columns(old_df, new_df)

    # See if the string columns (that are the boolean ones) have the same boolean representation
    compare_string_columns(old_df, new_df)

    # Get the same 'full' months for both dataframes, so the values are checked correspondingly
    old_df, new_df = filter_months(old_df, new_df, [11, 12, 1, 2])

    # Check the mean and std differences between the old and new data
    check_stats_within_magnitude(old_df, new_df)
