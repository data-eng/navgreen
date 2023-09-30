import pandas as pd
import numpy as np

df=pd.read_csv( "DATA_FROM_PLC.csv",
                parse_dates=["Date&time"],
                low_memory=False )
df.rename( columns={
        "3-WAY_EVAP_OPERATION": "THREE_WAY_EVAP_OPERATION",
        "3-WAY_COND_OPERATION": "THREE_WAY_COND_OPERATION",
        "3-WAY_SOLAR_OPERATION": "THREE_WAY_SOLAR_OPERATION",
        "4_WAY_VALVE": "FOUR_WAY_VALVE",
        "Date&time": "DATETIME" }, inplace=True )
print( df.shape )

# Marked as "not important"
df.drop( "SPARE_NTC_SENSOR", axis=1, inplace=True )
df.drop( "RECEIVER_LIQUID_IN", axis=1, inplace=True )
df.drop( "RECEIVER LIQUID_OUT", axis=1, inplace=True )
df.drop( "ECO_LIQUID_OUT", axis=1, inplace=True )
df.drop( "SUCTION_TEMP", axis=1, inplace=True )
df.drop( "DISCHARGE_TEMP", axis=1, inplace=True )
df.drop( "ECO_VAPOR_TEMP", axis=1, inplace=True )
df.drop( "EXPANSION_TEMP", axis=1, inplace=True )
df.drop( "ECO_EXPANSION_TEMP", axis=1, inplace=True )
df.drop( "POWER_GLOBAL_SOL", axis=1, inplace=True )
df.drop( "EEV_LOAD1", axis=1, inplace=True )
df.drop( "EEV_LOAD2", axis=1, inplace=True )

# Redundant
df.drop( "Date", axis=1, inplace=True )
df.drop( "Time", axis=1, inplace=True )

print( df.shape )
