import pandas as pd
import matplotlib.pyplot as plt
import logging

from navgreen_base import process_data, weather_parser_lp, create_weather_dataframe

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def plot_metrics(df):

    plot_df = pd.DataFrame({'binned_Q_PVT': []})
    plot_df['binned_Q_PVT'] = df['binned_Q_PVT'].dropna()
    plot_df[['binned_Q_PVT']] = plot_df[['binned_Q_PVT']].apply(pd.to_numeric)

    # Plot histogram
    plot_df.plot.hist(bins=range(6), edgecolor='black', alpha=0.7)
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title('Histogram of Q_PVT Categories')
    plt.xticks(range(5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("figures/binned_Q_PVT_hist.png", dpi=300)

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
    logger.info(f"HEATING at data df {len(df)} rows")

    # Drop columns that are not used
    for c in [c for c in df.columns if c not in keep_columns]:
        df.drop(c, axis="columns", inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Data df before {grp} aggregation: {len(df)} rows")
    # Group by grp intervals and apply different aggregations to each column
    df = df.groupby(pd.Grouper(key='DATETIME', freq=grp)).agg(aggregators)
    logger.info(f"Data df before {grp} aggregation: {len(df)} rows")

    df['Q_PVT'] = df['Q_PVT'] * 3
    df['POWER_PVT'] = df['POWER_PVT'] * 3

    # Define the bins and labels
    bins = [-float('inf'), 0.42, 1.05, 1.51, 2.14, float('inf')]
    labels = [0, 1, 2, 3, 4]

    # Add the binned Q_PVT based on the bins and labels
    df['binned_Q_PVT'] = pd.cut(df['Q_PVT'], bins=bins, labels=labels, right=False)

    # Make DATETIME a regular column and not index
    df = df.reset_index()

    return df


def combine_dataframes(weather_df, data_df):

    # Filter dataframes wrt the dates we care about
    logger.info(f"All data | Weather df: {len(weather_df)} rows & Data df: {len(data_df)} rows")
    weather_df = weather_df[(weather_df['DATETIME'] > '2022-08-31') & (weather_df['DATETIME'] < '2023-09-01')]
    data_df = data_df[(data_df['DATETIME'] > '2022-08-31') & (data_df['DATETIME'] < '2023-09-01')]
    logger.info(f"12 full months [2022-09 to 2023-08] | Weather df: {len(weather_df)} rows & Data df: {len(data_df)} rows")

    # Concatenate DataFrames along the 'DATETIME' column
    df = pd.concat([weather_df.set_index('DATETIME'), data_df.set_index('DATETIME')], axis=1, join='outer')

    return df



def main():
    group = "3h"
    # Parse weather data
    parsed_weather_data = weather_parser_lp(weather_file="data/obs_weather.txt")
    # Define weather columns aggregator
    aggregators_weather = {'humidity' : 'mean', 'pressure' : 'mean', 'feels_like' : 'mean', 'temp' : 'mean',
                           'wind_speed' : 'mean', 'rain_1h' : 'sum', 'rain_3h' : 'sum', 'snow_1h' : 'sum',
                           'snow_3h' : 'sum'}
    # Define weather DataFrame
    weather_df = create_weather_dataframe(parsed_weather_dict=parsed_weather_data,
                                          grp=group,
                                          aggregators=aggregators_weather)
    # Define data columns aggregator
    aggregators_data = {'OUTDOOR_TEMP': 'mean', 'PYRANOMETER': 'mean', 'DHW_BOTTOM': 'mean', 'POWER_PVT': 'mean',
                        'Q_PVT': 'mean'}

    # Columns that should not be filtered out
    data_columns_to_keep = ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT", "DATETIME"]
    data_df = create_data_dataframe(data_file="data/DATA_FROM_PLC.csv", keep_columns=data_columns_to_keep, grp=group,
                                    aggregators=aggregators_data)

    weather_df.to_csv('weather_df.csv', index=False)
    data_df.to_csv('data_df.csv', index=False)

    weather_df = pd.read_csv('weather_df.csv')
    data_df = pd.read_csv('data_df.csv')

    # Create the combined DataFrame
    combined_df = combine_dataframes(weather_df=weather_df, data_df=data_df)
    combined_df.to_csv('data/Q_PVT_classification_dataset.csv', index=False)

    # Plot some metrics
    plot_metrics(df=combined_df)

    # Show the remaining values after removing NaN rows
    combined_df.dropna(inplace=True)
    logger.info(combined_df)
