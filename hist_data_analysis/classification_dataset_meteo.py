import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

from navgreen_base import process_weather_df, create_weather_dataframe

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def plot_hist(df, y_name, data):

    plot_df = pd.DataFrame({y_name: []})
    plot_df[y_name] = df[y_name].dropna()
    plot_df[[y_name]] = plot_df[[y_name]].apply(pd.to_numeric)

    # Plot histogram
    plot_df.plot.hist(bins=range(6), edgecolor='black', alpha=0.7)
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Q_PVT Categories ({data} data)')
    plt.xticks(range(5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def train_test_split(df, train_path, test_path):
    # Convert the datetime column to datetime type if it's not already
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    init_columns = sorted(df.columns)

    # Calculate the day position relative to the end of the month
    df['days_from_end'] = df['DATETIME'].dt.days_in_month - df['DATETIME'].dt.day

    # Define a function to categorize the days
    def categorize_days(days):
        if days == 0:
            return 'last'
        elif days == 1:
            return 'penultimate'
        elif days == 2:
            return 'antepenultimate'
        else:
            return 'other'

    # Categorize the days
    df['day_category'] = df['days_from_end'].apply(lambda x: categorize_days(x))

    # Split the DataFrame into three separate DataFrames based on day category
    test_set = df[df['day_category'].isin(['last', 'penultimate', 'antepenultimate'])]
    training_set = df[df['day_category'] == 'other']

    # logger.info the length of each DataFrame
    logger.info(f"Length of Test Set: {len(test_set)}")
    logger.info(f"Length of Training Set: {len(training_set)}")

    # Assert that the sum of lengths matches the length of the original DataFrame
    assert (len(test_set) + len(training_set)) == len(df)

    # logger.info the number of different days within each dataset
    logger.info(f"Number of different days in Test Set: {test_set['DATETIME'].dt.day.nunique()}" )
    logger.info(f"Number of different days in Training Set: {training_set['DATETIME'].dt.day.nunique()}")

    # logger.info the number of unique dates within each dataset
    logger.info(f"Number of different dates in Test Set: {test_set['DATETIME'].dt.date.nunique()}")
    logger.info(f"Number of different dates in Training Set: {training_set['DATETIME'].dt.date.nunique()}")

    # Assert that the sum of unique days matches the one of the original DataFrame
    assert df['DATETIME'].dt.date.nunique() == (test_set['DATETIME'].dt.date.nunique() +
                                                training_set['DATETIME'].dt.date.nunique())

    # Drop unwanted columns
    test_set = test_set.drop(['days_from_end', 'day_category'], axis=1)
    training_set = training_set.drop(['days_from_end', 'day_category'], axis=1)

    assert (init_columns == sorted(training_set.columns)) and (init_columns == sorted(test_set.columns))

    # Save each DataFrame to a separate CSV file in the specified path
    test_set.to_csv(test_path, index=False)
    training_set.to_csv(train_path, index=False)

    return training_set, test_set


def create_data_dataframe(data_df, keep_columns, grp, aggregators, y_name, out_filename):

    if os.path.exists(out_filename):
        logger.info("Aggregated PLC data DataFrame exists. Loading it.")
        df = pd.read_csv(out_filename)
    else:
        logger.info(f"Creating aggregated PLC data DataFrame.")

        df = data_df

        df["Q_PVT"] = ((3.6014 + 0.004 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.000002 *
                        pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) *
                       ((1049.0 - 0.475 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.0018 *
                         pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) / 3600.0 * df["FLOW_PVT"]) *
                       (df["PVT_OUT"] - df["PVT_IN"]))

        # Heating mode
        df = df.loc[df["FOUR_WAY_VALVE"] == True]
        logger.info(f"HEATING at data DataFrame {len(df)} rows")

        # Drop columns that are not used
        keep_columns_ = keep_columns + ["Q_PVT"]
        df = df.drop([c for c in df.columns if c not in keep_columns_], axis=1)
        df = df.dropna()
        df = df.reset_index()

        df.rename(columns={"Date_time_local": "DATETIME"}, inplace=True)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])

        logger.info(f"Data df before {grp} aggregation: {len(df)} rows")
        # Group by grp intervals and apply different aggregations to each column
        df = df.groupby(pd.Grouper(key='DATETIME', freq=grp)).agg(aggregators)
        logger.info(f"Data df after {grp} aggregation: {len(df)} rows")

        df['Q_PVT'] = df['Q_PVT'] * 3

        # Define the bins and labels
        bins = [-float('inf'), 0.05, 0.21, 0.53, 1.05, float('inf')]
        labels = [0, 1, 2, 3, 4]

        # Add the binned Q_PVT based on the bins and labels
        df[y_name] = pd.cut(df['Q_PVT'], bins=bins, labels=labels, right=False)
        df = df.drop(['Q_PVT'], axis=1)

        # Make DATETIME a regular column and not index
        df = df.reset_index()

        df.to_csv(out_filename, index=False)

    return df


def combine_dataframes(weather_df, data_df, out_filename):

    if os.path.exists(out_filename):
        logger.info("Combined weather and data DataFrame exists. Loading it.")
        df = pd.read_csv(out_filename)
    else:
        logger.info(f"Creating combined weather and data DataFrame.")

        # Filter dataframes wrt the dates we care about
        logger.info(f"All data | Weather DataFrame: {len(weather_df)} rows & Data DataFrame: {len(data_df)} rows")

        weather_df['DATETIME'] = pd.to_datetime(weather_df['DATETIME'])
        data_df['DATETIME'] = pd.to_datetime(data_df['DATETIME'])

        weather_dates = weather_df['DATETIME']

        # Opt for days we have full weather observations, so discard the first and the last one
        start_date = weather_dates.min().date() + pd.Timedelta(days=1)
        end_date = weather_dates.max().date() - pd.Timedelta(days=1)

        # Filter DataFrames for the correct dates
        weather_df = weather_df[(weather_df['DATETIME'].dt.date >= start_date) & (weather_df['DATETIME'].dt.date <= end_date)]
        data_df = data_df[(data_df['DATETIME'].dt.date >= start_date) & (data_df['DATETIME'].dt.date <= end_date)]

        logger.info(f"Compatible dates | Weather DataFrame: {len(weather_df)} rows & Data DataFrame: {len(data_df)} rows")

        # Concatenate DataFrames along the 'DATETIME' column
        df = pd.concat([weather_df.set_index('DATETIME'), data_df.set_index('DATETIME')], axis=1, join='outer')
        logger.info(f"Combined DataFrame: {len(df)} rows")
        df = df.reset_index()

        df.to_csv(out_filename, index=False)

        # Show the remaining values after removing NaN rows
        # df.dropna(inplace=True)
        logger.info(f"Combined DataFrame after removing NaNs: {len(df.dropna())} rows")

    return df


def combine_PLC_data(in_path, out_filename):

    if os.path.exists(out_filename):
        logger.info("Concatenated PLC data DataFrame exists. Loading it.")
        df = pd.read_csv(out_filename)
    else:
        logger.info(f"Creating concatenated PLC data DataFrame.")

        dataframes = []

        for file_name in os.listdir(in_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(in_path, file_name)
                df = pd.read_csv(file_path)
                dataframes.append(df)

        df = pd.concat(dataframes, ignore_index=True)
        df['Date_time_local'] = pd.to_datetime(df['Date_time_local'], format='%Y-%m-%d %H:%M:%S')
        df = df.sort_values(by='Date_time_local')

        df.to_csv(out_filename, index=False)

    return df


def main():
    # Parse and process the weather DataFrame
    weather_df = create_weather_dataframe()
    weather_df = process_weather_df(df=weather_df, drop_columns=['SUNRISE', 'SUNSET'])

    data_df = combine_PLC_data(in_path='../data/data_plc', out_filename='../data/PLC_may_to_dec_2024.csv')

    # Define data columns aggregator
    group = "3h"
    aggregators_data = {'Q_PVT': 'mean'}

    pred_column = 'ValueRangeBin'

    # Columns that should not be filtered out
    data_columns_to_keep = ["Date_time_local"]
    data_df = create_data_dataframe(data_df=data_df,
                                    keep_columns=data_columns_to_keep,
                                    grp=group,
                                    aggregators=aggregators_data,
                                    out_filename=f'../data/PLC_may_to_dec_2024_aggr_{group}.csv',
                                    y_name=pred_column)

    # Create the combined data-weather DataFrame
    combined_df = combine_dataframes(weather_df=weather_df,
                                     data_df=data_df,
                                     out_filename='../data/combined_jun_to_dec_2024.csv')

    # Plot distribution of bins
    plot_hist(df=combined_df, y_name=pred_column, data='all')

    train_set, test_set = train_test_split(df=combined_df,
                                           test_path='../data/test_classif_meteo.csv',
                                           train_path='../data/train_classif_meteo.csv')

    plot_hist(df=train_set, y_name=pred_column, data='train')
    plot_hist(df=test_set, y_name=pred_column, data='test')


main()
