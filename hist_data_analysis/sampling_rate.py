import numpy as np
import pandas as pd
from hist_data_analysis import similarity


columns = ["PVT_IN_TO_DHW", "PVT_OUT_FROM_DHW",
           "PVT_IN_TO_SOLAR_BUFFER", "PVT_OUT_FROM_SOLAR_BUFFER",
           "SOLAR_BUFFER_IN", "SOLAR_BUFFER_OUT", "BTES_TANK_IN",
           "BTES_TANK_OUT", "SOLAR_HEAT_REJECTION_IN",
           "SOLAR_HEAT_REJECTION_OUT", "WATER_IN_EVAP",
           "WATER_OUT_EVAP", "WATER_IN_COND", "WATER_OUT_COND",
           "SH1_IN", "SH1_RETURN", "AIR_HP_TO_BTES_TANK",
           "DHW_INLET", "DHW_OUTLET", "DHW_BOTTOM", "SH_INLET",
           "SH_RETURN", "PVT_IN", "PVT_OUT"]

def load_csv():
    df = pd.read_csv( "data/DATA_FROM_PLC.csv",
                      parse_dates=["Date&time"], low_memory=False )
    df.rename(columns={
            "3-WAY_EVAP_OPERATION": "THREE_WAY_EVAP_OPERATION",
            "3-WAY_COND_OPERATION": "THREE_WAY_COND_OPERATION",
            "3-WAY_SOLAR_OPERATION": "THREE_WAY_SOLAR_OPERATION",
            "4_WAY_VALVE": "FOUR_WAY_VALVE",
            "Date&time": "DATETIME",
            "RECEIVER LIQUID_OUT": "RECEIVER_LIQUID_OUT"}, inplace=True)
    # Marked as "not important"
    df.drop("SPARE_NTC_SENSOR", axis=1, inplace=True)
    df.drop("POWER_GLOBAL_SOL", axis=1, inplace=True)

    # Redundant
    df.drop("Date", axis=1, inplace=True)
    df.drop("Time", axis=1, inplace=True)
    return df
#end def load_csv




def undersampling_error( t, v ):
    retv = pd.DataFrame(
        columns=["Undersampling Ratio", ])

    for i in range(3,8):
        step = 2**i
        #print( "  When keeping 1 out of {}".format(step) )
        sim = similarity( t, v, t[::step], v[::step] )
        sim["plot_means"].savefig( "means-{}.png".format(step) )
        sim["plot_slopes"].savefig( "slopes-{}.png".format(step) )
#end def undersampling_error


def runme():
    df = load_csv()
    dfQ4=df[(df['DATETIME'] > '2022-10-01') & (df['DATETIME'] < '2023-01-01')]
    for col in ["PVT_IN_TO_DHW"]: #columns:
        #print( "{} {}".format(col, df[col].isna().sum()) )
        err = undersampling_error( dfQ4["DATETIME"], dfQ4[col] )
        print( err )
    return df
