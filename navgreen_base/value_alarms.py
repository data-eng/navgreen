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


def load_data():
    """
    Loads the data from the historical data DataFrame
    :return: whole dataframe, dataframe for hp, dataframe for solar
    """
    df = pandas.read_csv("data/DATA_FROM_PLC.csv", parse_dates=["Date&time"], low_memory=False)
    df = navgreen_base.process_data(df, hist_data=True)
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
            if reldiffavg > 42.0:
                print("{} is having a bad month, rel diff {}".format(col, reldiffavg))
        for day in range(day00, day30 + 1):
            df_daily = df[df.DATETIME.dt.day == day]
            reldiffavg = df_daily[col].mean()/mavg - 1
            if reldiffavg < 0: reldiffavg = -reldiffavg
            if reldiffavg > 42.0:
                print( "{} is having a bad day, rel diff {}".format(col,reldiffavg) )
    dfretv = pandas.DataFrame([retv], columns=navgreen_base.numerical_columns)
    return dfretv
#end def month_stats

def main():
    df = load_data()
    dfavg = None
    for m in range(9, 21):
        df_monthly = df[df.DATETIME.dt.month == ((m - 1) % 12) + 1]
        logger.info("{} to {}".format(df_monthly.DATETIME.dt.date.iloc[0],
                                      df_monthly.DATETIME.dt.date.iloc[-1]))
        newavg = month_stats(df_monthly,dfavg)
        if dfavg is None: dfavg = newavg
        else: dfavg = (dfavg + newavg)/2
