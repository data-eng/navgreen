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

water_temp=["PVT_IN_TO_DHW","PVT_OUT_FROM_DHW","PVT_IN_TO_SOLAR_BUFFER",
            "PVT_OUT_FROM_SOLAR_BUFFER","SOLAR_BUFFER_IN","SOLAR_BUFFER_OUT",
            "BTES_TANK_IN","BTES_TANK_OUT","SOLAR_HEAT_REJECTION_IN",
            "SOLAR_HEAT_REJECTION_OUT","WATER_IN_EVAP","WATER_OUT_EVAP",
            "WATER_IN_COND","WATER_OUT_COND","SH1_IN","SH1_RETURN",
            "AIR_HP_TO_BTES_TANK","DHW_INLET","DHW_OUTLET","DHW_BOTTOM",
            "SH_INLET","SH_RETURN","PVT_IN","PVT_OUT"]

other_temp=["OUTDOOR_TEMP","BTES_TANK","SOLAR_BUFFER_TANK",
            "SH_BUFFER","DHW_BUFFER","INDOOR_TEMP"]

pressure=["SUCTION_PRESSURE","DISCHARGE_PRESSURE","ECO_PRESSURE"]

flow=["FLOW_EVAPORATOR","FLOW_CONDENSER","FLOW_DHW",
      "FLOW_SOLAR_HEAT_REJECTION","FLOW_PVT","FLOW_FAN_COILS_INDOOR"]

other=["POWER_HP","POWER_PVT","PYRANOMETER","Compressor_HZ"]

control=["THREE_WAY_EVAP_OPERATION","THREE_WAY_COND_OPERATION",
         "THREE_WAY_SOLAR_OPERATION","FOUR_WAY_VALVE",
         "AIR_COOLED_COMMAND","Residential_office_mode","MODBUS_LOCAL"]

# just checking that nothing is fogotten. +1 for datetime
assert len(df.columns) == len(water_temp)+len(other_temp)+len(pressure)+len(flow)+len(other)+len(control)+1

for col in other_temp:
    for offset in [1,5,10]:
        timestep = df.iloc[offset,0] - df.iloc[0,0]
        dfPrev = df.shift( offset )
        consec = ( df.DATETIME - dfPrev.DATETIME <= 2*timestep )
        delta = np.abs(df[col]-dfPrev[col])
        print( "Describe delta {} at offset {}:".format(col,offset) )
        print( delta.describe() )

#df.to_csv("aa.csv")
