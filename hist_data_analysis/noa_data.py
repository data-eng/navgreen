import pandas as pd
import matplotlib.pyplot as plt
import numpy
import logging

from navgreen_base import process_data, weather_parser_lp, create_weather_dataframe

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def makeFacet(df, facetname, thresholds):
    assert len(thresholds) == 4
    conditions = [
        pd.isnull(df.Q_PVT),
        (df["Q_PVT"] <= thresholds[0]),
        (df["Q_PVT"] < thresholds[1]) & (df["Q_PVT"] >= thresholds[0]),
        (df["Q_PVT"] < thresholds[2]) & (df["Q_PVT"] >= thresholds[1]),
        (df["Q_PVT"] < thresholds[3]) & (df["Q_PVT"] >= thresholds[2]),
        (df["Q_PVT"] >= thresholds[3])
    ]
    values = [numpy.nan, 0, 1, 2, 3, 4]
    df[facetname] = numpy.select(conditions, values)
    return df


def classCount(df):
    df = df[pd.notnull(df.Q_PVT)]
    df_daytime = df[(df.DATETIME.dt.hour > 5) & (df.DATETIME.dt.hour < 19)]
    print(df_daytime.Q_PVT.max())
    sorted_ = df_daytime.Q_PVT.sort_values().to_numpy()

    # index where values > 0.05 start
    gt05 = (numpy.argwhere(sorted_ > 0.05))[0][0]
    # step = (len(sorted) - gt05)
    idx = numpy.arange(gt05, len(sorted_)-1, 208.5).astype(int)

    print(sorted_[idx])
    df_out = df.copy()
    for v in sorted_[idx]:
        df_in = df_out[df["Q_PVT"] < v]
        df_out = df[df["Q_PVT"] >= v]
        print("{} rows less than {}, {} rows remainder".format(len(df_in), v, len(df_out)))

    return sorted_[idx]


def binnings(df_train_path, df_test_path):

    df = pd.read_csv(df_train_path, parse_dates=["DATETIME"])
    thresholds = {}

    facetname = "FixedBin"
    print(facetname)
    thresholds[facetname] = [0.05, 0.5, 1.0, 1.5]
    df = makeFacet(df, facetname, thresholds[facetname])
    print("{} class sizes, training set: ".format(facetname))
    print(df[facetname].value_counts())

    facetname = "ValueCountBin"
    print(facetname)
    thresholds[facetname] = [0., 2., 3., 4.]  # classCount(df[["DATETIME", "Q_PVT"]])
    df = makeFacet(df, facetname, thresholds[facetname])
    print("{} class sizes, training set: ".format(facetname))
    print(df[facetname].value_counts())

    facetname = "ValueRangeBin"
    print(facetname)
    thresholds[facetname] = [0.05, 0.21, 0.525, 1.05]
    df = makeFacet(df, facetname, thresholds[facetname])
    print("{} class sizes, training set: ".format(facetname))
    print(df[facetname].value_counts())

    nancount = len(df[pd.isna(df["Q_PVT"])])
    notnancount = df.ValueCountBin.value_counts().sum()
    assert nancount + notnancount == len(df)

    df.to_csv(df_train_path)

    df = pd.read_csv(df_test_path, parse_dates=["DATETIME"])

    facetname = "FixedBin"
    print(facetname)
    df = makeFacet(df, facetname, thresholds[facetname])
    print("{} class sizes, test set: ".format(facetname))
    print(df[facetname].value_counts())

    facetname = "ValueCountBin"
    print(facetname)
    df = makeFacet(df, facetname, thresholds[facetname])
    print("{} class sizes, test set: ".format(facetname))
    print(df[facetname].value_counts())

    facetname = "ValueRangeBin"
    print(facetname)
    df = makeFacet(df, facetname, thresholds[facetname])
    print("{} class sizes, test set: ".format(facetname))
    print(df[facetname].value_counts())

    nancount = len(df[pd.isna(df["Q_PVT"])])
    notnancount = df.ValueCountBin.value_counts().sum()
    assert nancount + notnancount == len(df)

    df.to_csv(df_test_path)


def train_test_split(data_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(data_path)
    # Convert the datetime column to datetime type if it's not already
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    # Split with new dates
    # Train data before March-2023
    training_set = df[df['DATETIME'] < '2023-03-01']
    # Test data from March-2023 to December-2023
    test_set = df[(df['DATETIME'] > '2023-03-01') & (df['DATETIME'] < '2024-01-01')]

    # Save each DataFrame to a separate CSV file in the specified path
    train_df_pth = '../data/training_set_classif_new_classes_noa.csv'
    test_df_pth = '../data/test_set_classif_new_classes_noa.csv'
    training_set.to_csv(train_df_pth, index=False)
    test_set.to_csv(test_df_pth, index=False)

    # Create the bins
    binnings(df_train_path=train_df_pth, df_test_path=test_df_pth)


def create_data_dataframe(data_file, keep_columns, grp, aggregators):

    df = pd.read_csv(data_file, parse_dates=["Date&time"], low_memory=False)
    df = process_data(df, hist_data=True)

    df["Q_PVT"] = ((3.6014 + 0.004 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.000002 *
                    pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) *
                   ((1049.0 - 0.475 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.0018 *
                     pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) / 3600.0 * df["FLOW_PVT"]) *
                   (df["PVT_OUT"] - df["PVT_IN"]))

    # Heating mode
    df = df.loc[df["FOUR_WAY_VALVE"] == "HEATING"]
    # df = df.loc[(df["FOUR_WAY_VALVE"] == "HEATING") | (df["FOUR_WAY_VALVE"] == "1")]
    logger.info(f"HEATING at data df {len(df)} rows")

    # Drop columns that are not used
    for c in [c for c in df.columns if c not in keep_columns]:
        df.drop(c, axis="columns", inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Data df before {grp} aggregation: {len(df)} rows")
    # Group by grp intervals and apply different aggregations to each column
    df = df.groupby(pd.Grouper(key='DATETIME', freq=grp)).agg(aggregators)
    logger.info(f"Data df after {grp} aggregation: {len(df)} rows")

    # Make DATETIME a regular column and not index
    df = df.reset_index()

    return df


def create_navgreen_data_dataframe(data_file, keep_columns, grp, aggregators):

    df = pd.read_csv(data_file, parse_dates=["DATETIME"], low_memory=False)

    df["Q_PVT"] = ((3.6014 + 0.004 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.000002 *
                    pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) *
                   ((1049.0 - 0.475 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.0018 *
                     pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) / 3600.0 * df["FLOW_PVT"]) *
                   (df["PVT_OUT"] - df["PVT_IN"]))

    # Heating mode
    df = df.loc[(df["FOUR_WAY_VALVE"] == "True") | (df["FOUR_WAY_VALVE"] == "1.0")]
    logger.info(f"HEATING at data df {len(df)} rows")

    # Drop columns that are not used
    for c in [c for c in df.columns if c not in keep_columns]:
        df.drop(c, axis="columns", inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Data df before {grp} aggregation: {len(df)} rows")
    # Group by grp intervals and apply different aggregations to each column
    df = df.groupby(pd.Grouper(key='DATETIME', freq=grp)).agg(aggregators)
    logger.info(f"Data df after {grp} aggregation: {len(df)} rows")

    # Make DATETIME a regular column and not index
    df = df.reset_index()

    return df


def create_navgreen_noa_data_dataframe(noa_df_pth, grp, aggregators):
    df = pd.read_csv(noa_df_pth, low_memory=False)
    df['DATETIME'] = pd.to_datetime(df['DAY'] + ' ' + df['TIME'], format='%d/%m/%Y %H:%M')
    df = df.drop(columns=['DAY', 'TIME'])

    logger.info(f"NOA df before {grp} aggregation: {len(df)} rows")
    # Group by grp intervals and apply different aggregations to each column
    df = df.groupby(pd.Grouper(key='DATETIME', freq=grp)).agg(aggregators)
    logger.info(f"NOA df after {grp} aggregation: {len(df)} rows")

    # Make DATETIME a regular column and not index
    df = df.reset_index()

    return df


def combine_dataframes(weather_df, data_df):

    # Filter dataframes wrt the dates we care about
    logger.info(f"All data | df1: {len(weather_df)} rows & df2: {len(data_df)} rows")
    # Concatenate DataFrames along the 'DATETIME' column
    df = pd.concat([weather_df.set_index('DATETIME'), data_df.set_index('DATETIME')], axis=1, join='outer')
    logger.info(f"Concat data : {len(df)}")

    return df


def concatenate_and_sort(df1, df2, column='DATETIME'):
    logger.info(f"df1 : {len(df1)} rows & new df2: {len(df2)} rows")

    df1['DATETIME'] = pd.to_datetime(df1['DATETIME'], utc=True)
    df2['DATETIME'] = pd.to_datetime(df2['DATETIME'], utc=True)

    # Concatenate the two DataFrames
    concatenated_df = pd.concat([df1, df2])
    # Sort the concatenated DataFrame based on the datetime_column
    sorted_df = concatenated_df.sort_values(by=column)
    # Convert the timestamps back to a timezone-naive representation
    sorted_df['DATETIME'] = sorted_df['DATETIME'].dt.tz_convert(None)

    logger.info(f"Together : {len(sorted_df)}")

    return sorted_df


def open_weather_and_installation_data():

    group = "1h"
    # Parse weather data
    parsed_weather_data = weather_parser_lp(weather_file="../data/obs_weather.txt")
    # Define weather columns aggregator
    aggregators_weather = {'humidity': 'mean', 'pressure': 'mean', 'feels_like': 'mean', 'temp': 'mean',
                           'wind_speed': 'mean', 'rain_1h': 'sum', 'rain_3h': 'sum', 'snow_1h': 'sum',
                           'snow_3h': 'sum'}
    # Define weather DataFrame
    weather_df = create_weather_dataframe(parsed_weather_dict=parsed_weather_data,
                                          grp=group,
                                          aggregators=aggregators_weather)

    # Define data columns aggregator
    aggregators_data = {'OUTDOOR_TEMP': 'mean', 'PYRANOMETER': 'mean', 'DHW_BOTTOM': 'mean', 'POWER_PVT': 'mean',
                        'Q_PVT': 'mean'}

    # Columns that should not be filtered out
    data_columns_to_keep = ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT", "DATETIME"]

    hist_df = create_data_dataframe(data_file="../data/DATA_FROM_PLC.csv", keep_columns=data_columns_to_keep, grp=group,
                                    aggregators=aggregators_data)

    # Read the data from the NavGreen Project
    nav_df = create_navgreen_data_dataframe(data_file="../data/concatenated_data.csv",
                                            keep_columns=data_columns_to_keep, grp=group,
                                            aggregators=aggregators_data)

    # Concatenate all of our available data
    data_df = concatenate_and_sort(hist_df, nav_df)

    # Get NOA weather data
    aggregators_noa = {'T2': 'mean', 'RH2': 'mean', 'WSPD': 'mean', 'WDIR': 'mean', 'RAIN': 'sum'}
    noa_df = create_navgreen_noa_data_dataframe(noa_df_pth='../data/North_Athens_2023_data.csv',
                                                grp=group, aggregators=aggregators_noa)

    weather_df.to_csv('../data/weather_df.csv', index=False)
    data_df.to_csv('../data/data_df.csv', index=False)
    noa_df.to_csv('../data/noa_df.csv', index=False)

    weather_df = pd.read_csv('../data/weather_df.csv')
    data_df = pd.read_csv('../data/data_df.csv')
    noa_df = pd.read_csv('../data/noa_df.csv')

    # Create the combined DataFrame
    combined_df = combine_dataframes(weather_df=weather_df, data_df=data_df)
    combined_df = combined_df.reset_index()
    combined_df = combined_df.sort_values(by='DATETIME')
    combined_df = combine_dataframes(weather_df=noa_df, data_df=combined_df)
    combined_df = combined_df.sort_values(by='DATETIME')
    combined_df.to_csv('../data/Q_PVT_classification_dataset.csv', index=True)

    # Show the remaining values after removing NaN rows
    combined_df.dropna(inplace=True)
    logger.info(combined_df.shape)

    train_test_split('../data/Q_PVT_classification_dataset.csv')


open_weather_and_installation_data()