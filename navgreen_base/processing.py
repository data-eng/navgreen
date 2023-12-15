import pandas as pd
import numpy as np


columns=['Date_time_local', 'DATETIME', 'PVT_IN_TO_DHW', 'PVT_OUT_FROM_DHW', 'PVT_IN_TO_SOLAR_BUFFER',
     'PVT_OUT_FROM_SOLAR_BUFFER', 'SOLAR_BUFFER_IN', 'SOLAR_BUFFER_OUT', 'BTES_TANK_IN',
     'BTES_TANK_OUT', 'SOLAR_HEAT_REJECTION_IN', 'SOLAR_HEAT_REJECTION_OUT',
     '3-WAY_EVAP_OPERATION', '3-WAY_COND_OPERATION', '3-WAY_SOLAR_OPERATION', 'SPARE_NTC_SENSOR',
     'Date', 'Time', 'RECEIVER_LIQUID_IN', 'RECEIVER_LIQUID_OUT', 'ECO_LIQUID_OUT',
     'SUCTION_TEMP', 'DISCHARGE_TEMP', 'ECO_VAPOR_TEMP', 'EXPANSION_TEMP', 'ECO_EXPANSION_TEMP',
     'SUCTION_PRESSURE', 'DISCHARGE_PRESSURE', 'ECO_PRESSURE', 'AIR_COOLED_COMMAND',
     '4_WAY_VALVE', 'WATER_IN_EVAP', 'WATER_OUT_EVAP', 'WATER_IN_COND', 'WATER_OUT_COND',
     'OUTDOOR_TEMP', 'BTES_TANK', 'SOLAR_BUFFER_TANK', 'SH_BUFFER', 'DHW_BUFFER', 'INDOOR_TEMP',
     'DHW_INLET', 'DHW_OUTLET', 'SH1_IN', 'SH1_RETURN', 'DHW_BOTTOM', 'AIR_HP_TO_BTES_TANK',
     'SH_INLET', 'SH_RETURN', 'PVT_IN', 'PVT_OUT', 'POWER_HP', 'POWER_GLOBAL_SOL', 'POWER_PVT',
     'FLOW_EVAPORATOR', 'FLOW_CONDENSER', 'FLOW_DHW', 'FLOW_SOLAR_HEAT_REJECTION', 'FLOW_PVT',
     'FLOW_FAN_COILS_INDOOR', 'PYRANOMETER', 'Compressor_HZ', 'Residential_office_mode',
     'MODBUS_LOCAL', 'EEV_LOAD1', 'EEV_LOAD2', "T_CHECKPOINT_DHW_MODBUS", "T_CHECKPOINT_SPACE_HEATING_MODBUS"]


water_temp = ["PVT_IN_TO_DHW", "PVT_OUT_FROM_DHW", "PVT_IN_TO_SOLAR_BUFFER", "PVT_OUT_FROM_SOLAR_BUFFER",
              "SOLAR_BUFFER_IN", "SOLAR_BUFFER_OUT", "BTES_TANK_IN", "BTES_TANK_OUT", "SOLAR_HEAT_REJECTION_IN",
              "SOLAR_HEAT_REJECTION_OUT", "WATER_IN_EVAP", "WATER_OUT_EVAP", "WATER_IN_COND", "WATER_OUT_COND",
              "SH1_IN", "SH1_RETURN", "AIR_HP_TO_BTES_TANK", "DHW_INLET", "DHW_OUTLET", "DHW_BOTTOM", "SH_INLET",
              "SH_RETURN", "PVT_IN", "PVT_OUT"]

other_temp = ["OUTDOOR_TEMP", "BTES_TANK", "SOLAR_BUFFER_TANK", "SH_BUFFER", "DHW_BUFFER", "INDOOR_TEMP"]

ref_temp = ["RECEIVER_LIQUID_IN", "RECEIVER_LIQUID_OUT", "ECO_LIQUID_OUT", "SUCTION_TEMP",
            "DISCHARGE_TEMP", "ECO_VAPOR_TEMP", "EXPANSION_TEMP", "ECO_EXPANSION_TEMP"]

pressure = ["SUCTION_PRESSURE", "DISCHARGE_PRESSURE", "ECO_PRESSURE"]

flow = ["FLOW_EVAPORATOR", "FLOW_CONDENSER", "FLOW_DHW", "FLOW_SOLAR_HEAT_REJECTION",
        "FLOW_PVT", "FLOW_FAN_COILS_INDOOR"]

power = ["POWER_HP", "POWER_PVT"]

solar = ["PYRANOMETER"]

other = ["Compressor_HZ", "EEV_LOAD1", "EEV_LOAD2"]

control = ["THREE_WAY_EVAP_OPERATION", "THREE_WAY_COND_OPERATION", "THREE_WAY_SOLAR_OPERATION",
           "FOUR_WAY_VALVE", "AIR_COOLED_COMMAND", "Residential_office_mode", "MODBUS_LOCAL"]

checkpoints = ["T_CHECKPOINT_DHW_MODBUS", "T_CHECKPOINT_SPACE_HEATING_MODBUS"]

temp_sensors = []
temp_sensors.extend(water_temp)
temp_sensors.extend(other_temp)
temp_sensors.extend(ref_temp)


def process_data(df, hist_data=True):
    """
    This function applies the needed transformations to the columns of the input DataFrame and to the data outliers.
    :param df: DataFrame to be processed
    :param hist_data: Boolean parameter that indicated whether the input DataFrame refers to the historical data (below
    is explained why this is needed)
    :return: The processed DataFrame
    """

    # Rename some columns. Note that when these columns are not in the DataFrame no error is raised and of course the
    # renaming is not performed.
    df.rename(columns={
        "3-WAY_EVAP_OPERATION": "THREE_WAY_EVAP_OPERATION",
        "3-WAY_COND_OPERATION": "THREE_WAY_COND_OPERATION",
        "3-WAY_SOLAR_OPERATION": "THREE_WAY_SOLAR_OPERATION",
        "4_WAY_VALVE": "FOUR_WAY_VALVE",
        "Date&time": "DATETIME",
        "RECEIVER LIQUID_OUT": "RECEIVER_LIQUID_OUT"}, inplace=True)

    # Columns marked as "not important"
    columns_to_drop = ["SPARE_NTC_SENSOR", "POWER_GLOBAL_SOL"]
    # Redundant columns
    columns_to_drop += ["Date", "Time"]
    # For actual time
    columns_to_drop += ["Date_time_local"]

    # Check if these columns exist and then drop them. This check is needed as the input DataFrame might not be
    # consistent in that matter as it may come from different sources (historical, live and other combinations)
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # The historical data in our disposal (the ones that refer to the data collected before we took over the project)
    # were not converted to the preferred units, so when these data are used, the following conversions should be made
    if hist_data:
        # Convert to correct units
        columns_div10 = ["WATER_OUT_COND", "WATER_IN_COND", "WATER_IN_EVAP", "WATER_OUT_EVAP", "OUTDOOR_TEMP", "BTES_TANK",
                         "SOLAR_BUFFER_TANK", "SH_BUFFER", "DHW_BUFFER", "INDOOR_TEMP", "DHW_OUTLET", "DHW_INLET", "SH1_IN",
                         "SH1_RETURN", "SH_INLET", "SH_RETURN", "PVT_IN", "PVT_OUT", "PVT_IN_TO_DHW", "PVT_OUT_FROM_DHW",
                         "PVT_IN_TO_SOLAR_BUFFER", "PVT_OUT_FROM_SOLAR_BUFFER", "SOLAR_BUFFER_IN", "SOLAR_BUFFER_OUT",
                         "BTES_TANK_IN", "BTES_TANK_OUT"]

        columns_div1000 = ["POWER_HP"]

        columns_div10000 = ["FLOW_CONDENSER", "FLOW_EVAPORATOR", "FLOW_DHW", "FLOW_FAN_COILS_INDOOR", "POWER_PVT",
                            "PYRANOMETER", "FLOW_PVT"]

        '''
        # Measurements we do not read from the PLC yet
        columns_nan = ["SOLAR_HEAT_REJECTION_IN", "SOLAR_HEAT_REJECTION_OUT", "AIR_HP_TO_BTES_TANK", "DHW_BOTTOM",
                       "RECEIVER_LIQUID_IN", "RECEIVER_LIQUID_OUT", "ECO_LIQUID_OUT", "SUCTION_TEMP", "DISCHARGE_TEMP",
                       "ECO_VAPOR_TEMP", "EXPANSION_TEMP", "ECO_EXPANSION_TEMP", "SUCTION_PRESSURE",
                       "DISCHARGE_PRESSURE",
                       "ECO_PRESSURE", "FLOW_SOLAR_HEAT_REJECTION", "Compressor_HZ", "EEV_LOAD1", "EEV_LOAD2"]
        '''
        for col in columns_div10:
            df[col] = df[col] / 10

        for col in columns_div1000:
            df[col] = df[col] / 1000

        for col in columns_div10000:
            df[col] = df[col] / 10000

        '''
        for col in columns_nan:
            df[col] = np.nan
        '''

        df["DATETIME"] = pd.to_datetime(df['DATETIME'], format="%Y-%m-%d %H:%M:%S")

    # Apply reasonable limits
    for col in solar:
        df[col] = df[col].apply(lambda x: 0.0 if x > 2.0 else x)
    for col in temp_sensors:
        df[col] = df[col].apply(lambda x: np.nan if x < -20.0 or x > 100.0 else x)
    for col in pressure:
        df[col] = df[col].apply(lambda x: np.nan if x <  0.0 or x > 35.0 else x)

    df['FLOW_CONDENSER'] = df['FLOW_CONDENSER'].apply(lambda x: 0.0 if x >= 4.0 else x)

    return df
#end def process_data