from pymodbus.client import ModbusTcpClient

import pandas as pd
import numpy as np
import os

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime


def make_point(measurement, row, value_columns):
    p = influxdb_client.Point(measurement)
    p.time(row["DATETIME"])
    '''
    # Tag with the state of the valves, as context
    for col in tag_columns:
        if row[col] is not np.nan:
            p.tag(col, row[col])
    # Add the sensor data fields
    '''
    for col in value_columns:
        if row[col] is not np.nan:
            p.field(col, row[col])
        else:
            p.field(col, 9999999.9)
    return p


def delete_data(url, auth_token, organization):
    influx_client = influxdb_client.InfluxDBClient(url=url, token=auth_token, org=organization)
    api = influx_client.delete_api()
    api.delete(bucket="test_bucket", org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="temporary"')

    print("Deleted ok")


def write_data(row, url, token, organization):
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.write_api(write_options=SYNCHRONOUS)

    p = make_point("temporary", row, fields)
    api.write(bucket="test_bucket", org=organization, record=p)
    print("Write ok")


def read_data(url, token, organization):
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.query_api()
    query = 'from(bucket: "test_bucket") |> range(start: 0) |> filter(fn: (r) => r._measurement == "temporary")'
    data = api.query(org=organization, query=query)
    print("Read ok")

    # Extract the records from the result
    records = []
    for table in data:
        for record in table.records:
            records.append(record.values)

    # Create a list to store the unique "field" values
    unique_names = list(set(record["_field"] for record in records))
    d = dict()

    # Add a column for each unique "field" value
    for name in unique_names:
        name_data = [record["_value"] for record in records if record["_field"] == name]
        d[name] = name_data

    df = pd.DataFrame(d)

    return df


# Create Pandas DataFrame with the corresponding columns
golden_df = pd.DataFrame(columns=['DATETIME', 'PVT_IN_TO_DHW', 'PVT_OUT_FROM_DHW',
                                  'PVT_IN_TO_SOLAR_BUFFER', 'PVT_OUT_FROM_SOLAR_BUFFER', 'SOLAR_BUFFER_IN',
                                  'SOLAR_BUFFER_OUT', 'BTES_TANK_IN', 'BTES_TANK_OUT', 'SOLAR_HEAT_REJECTION_IN',
                                  'SOLAR_HEAT_REJECTION_OUT', '3-WAY_EVAP_OPERATION', '3-WAY_COND_OPERATION',
                                  '3-WAY_SOLAR_OPERATION', 'SPARE_NTC_SENSOR', 'RECEIVER_LIQUID_IN',
                                  'RECEIVER_LIQUID_OUT', 'ECO_LIQUID_OUT', 'SUCTION_TEMP', 'DISCHARGE_TEMP',
                                  'ECO_VAPOR_TEMP', 'EXPANSION_TEMP', 'ECO_EXPANSION_TEMP', 'SUCTION_PRESSURE',
                                  'DISCHARGE_PRESSURE', 'ECO_PRESSURE', 'AIR_COOLED_COMMAND', '4_WAY_VALVE',
                                  'WATER_IN_EVAP', 'WATER_OUT_EVAP', 'WATER_IN_COND', 'WATER_OUT_COND', 'OUTDOOR_TEMP',
                                  'BTES_TANK', 'SOLAR_BUFFER_TANK', 'SH_BUFFER', 'DHW_BUFFER', 'INDOOR_TEMP',
                                  'DHW_INLET', 'DHW_OUTLET', 'SH1_IN', 'SH1_RETURN', 'DHW_BOTTOM',
                                  'AIR_HP_TO_BTES_TANK', 'SH_INLET', 'SH_RETURN', 'PVT_IN', 'PVT_OUT', 'POWER_HP',
                                  'POWER_GLOBAL_SOL', 'POWER_PVT', 'FLOW_EVAPORATOR', 'FLOW_CONDENSER', 'FLOW_DHW',
                                  'FLOW_SOLAR_HEAT_REJECTION', 'FLOW_PVT', 'FLOW_FAN_COILS_INDOOR', 'PYRANOMETER',
                                  'Compressor_HZ', 'Residential_office_mode', 'MODBUS_LOCAL', 'EEV_LOAD1', 'EEV_LOAD2'])

golden_df.rename(columns={
    "3-WAY_EVAP_OPERATION": "THREE_WAY_EVAP_OPERATION",
    "3-WAY_COND_OPERATION": "THREE_WAY_COND_OPERATION",
    "3-WAY_SOLAR_OPERATION": "THREE_WAY_SOLAR_OPERATION",
    "4_WAY_VALVE": "FOUR_WAY_VALVE"}, inplace=True)

# Marked as "not important"
golden_df.drop("SPARE_NTC_SENSOR", axis=1, inplace=True)
golden_df.drop("POWER_GLOBAL_SOL", axis=1, inplace=True)

print("Finished preprocessing, shape: {}".format(golden_df.shape))

fields = ["PVT_IN_TO_DHW", "PVT_OUT_FROM_DHW", "PVT_IN_TO_SOLAR_BUFFER", "PVT_OUT_FROM_SOLAR_BUFFER", "SOLAR_BUFFER_IN",
          "SOLAR_BUFFER_OUT", "BTES_TANK_IN", "BTES_TANK_OUT", "SOLAR_HEAT_REJECTION_IN", "SOLAR_HEAT_REJECTION_OUT",
          "WATER_IN_EVAP", "WATER_OUT_EVAP", "WATER_IN_COND", "WATER_OUT_COND", "SH1_IN", "SH1_RETURN",
          "AIR_HP_TO_BTES_TANK", "DHW_INLET", "DHW_OUTLET", "DHW_BOTTOM", "SH_INLET", "SH_RETURN", "PVT_IN", "PVT_OUT",
          "OUTDOOR_TEMP", "BTES_TANK", "SOLAR_BUFFER_TANK", "SH_BUFFER", "DHW_BUFFER", "INDOOR_TEMP",
          "RECEIVER_LIQUID_IN", "RECEIVER_LIQUID_OUT", "ECO_LIQUID_OUT", "SUCTION_TEMP",  "DISCHARGE_TEMP",
          "ECO_VAPOR_TEMP", "EXPANSION_TEMP", "ECO_EXPANSION_TEMP", "SUCTION_PRESSURE", "DISCHARGE_PRESSURE",
          "ECO_PRESSURE", "FLOW_EVAPORATOR", "FLOW_CONDENSER", "FLOW_DHW", "FLOW_SOLAR_HEAT_REJECTION", "FLOW_PVT",
          "FLOW_FAN_COILS_INDOOR", "POWER_HP", "POWER_PVT", "PYRANOMETER", "Compressor_HZ", "EEV_LOAD1", "EEV_LOAD2",
          "THREE_WAY_EVAP_OPERATION", "THREE_WAY_COND_OPERATION", "THREE_WAY_SOLAR_OPERATION",  "FOUR_WAY_VALVE",
          "AIR_COOLED_COMMAND", "Residential_office_mode", "MODBUS_LOCAL", "DATETIME"]

print(f'Length of total fields: {len(fields)}.')


# CREDENTIALS: CAREFUL
url = os.environ.get('Url_influx_db')
org = os.environ.get('Organization_influx')
auth_token = os.environ.get('Auth_token')

# PLC IP address and port
plc_ip = os.environ.get('Plc_ip')
plc_port = 502
reconnect_interval = 10  # Seconds

dataframe_date = datetime.utcnow()
dataframe_date = dataframe_date.strftime('%Y-%m-%d')

client = None
try:
    # Create a Modbus TCP/IP client
    client = ModbusTcpClient(plc_ip, port=plc_port)

    # Try to establish a connection
    if client.connect():
        print("Connected to PLC successfully!")

        # Read holding register D500, D501, D502
        result_coils2 = client.read_coils(8192 + 509, 7, int=0)  # read coils from 509 to 515
        result = client.read_holding_registers(500, 20)  # Read registers from 500 to 519
        result_coils = client.read_coils(8192 + 500, 4)  # Read coils from 500 to 503
        results_registers2 = client.read_holding_registers(520, 18)  # Read registers from 520 to 527

        if result.isError():
            print("Error reading register.")
        else:
            T_cond_out = result.registers[0]
            T_cond_in = result.registers[1]
            flow_condenser = result.registers[2]
            T_evap_out = result.registers[3]
            T_evap_in = result.registers[4]
            flow_evap = result.registers[5]
            T_air_source = result.registers[6]
            T_BTES_source = result.registers[7]
            T_solar_buffer_source = result.registers[8]
            T_space_heating_buffer = result.registers[9]
            T_DHW_buffer = result.registers[10]
            T_indoor_temp = result.registers[11]
            T_dhw_out = result.registers[12]
            T_dhw_in = result.registers[13]
            flow_dhw = result.registers[14]
            T_hp_out_sh_in = result.registers[15]
            T_sh_out_hp_in = result.registers[16]
            T_hp_out_to_dhw_tank = result.registers[17]
            T_hp_in_from_dhw_tank = result.registers[18]
            Total_electric_power = result.registers[19]

            Air_source_valve = result_coils.bits[0]
            BTES_source_valve = result_coils.bits[1]
            PVT_setting_valve = result_coils.bits[2]
            Condenser_three_way = result_coils.bits[3]

            T_from_sh_to_demand = results_registers2.registers[0]
            T_from_demand_to_space_bufer = results_registers2.registers[1]
            flow_water_demand = results_registers2.registers[2]
            Power_PV = results_registers2.registers[3]
            Solar_irr_tilted = results_registers2.registers[4]
            T_pvt_in = results_registers2.registers[5]
            T_pvt_out = results_registers2.registers[6]
            T_pvt_in_to_dhw = results_registers2.registers[7]
            T_dhw_out_to_pvt = results_registers2.registers[8]
            T_pvt_in_to_solar_buffer = results_registers2.registers[9]
            T_solar_buffer_out_to_pvt = results_registers2.registers[10]
            flow_solar_circuit = results_registers2.registers[11]
            T_hp_out_to_solar_buffer_in = results_registers2.registers[12]
            T_hp_in_from_solar_buffer = results_registers2.registers[13]
            T_hp_out_to_btes_in = results_registers2.registers[14]
            T_BTES_out_to_hp_in = results_registers2.registers[15]
            T_setpoint_DHW_modbus = results_registers2.registers[16]
            T_setpoint_SPACE_HEATING_modbus = results_registers2.registers[17]

            Residential_office = result_coils2.bits[0]
            BTES_HEATING_DHW_THREE_WAY = result_coils2.bits[1]
            BTES_GROUND_SOLAR_VALVE = result_coils2.bits[2]
            BTES_SOLAR_THREE_WAY_VALVE = result_coils2.bits[3]
            BTES_WATER_AIR_OPERATION = result_coils2.bits[4]
            HEATING_COOLING_MODE = result_coils2.bits[5]
            BMES_LOCAL_CONTROL = result_coils2.bits[6]

            current_datetime = datetime.utcnow()
            current_date = current_datetime.strftime('%Y-%m-%d')
            current_datetime = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

            new_row = {"DATETIME": current_datetime,  # "DATETIME": pd.to_datetime(current_datetime),
                       "PVT_IN_TO_DHW": T_pvt_in_to_dhw,
                       "PVT_OUT_FROM_DHW": T_dhw_out_to_pvt,
                       "PVT_IN_TO_SOLAR_BUFFER": T_pvt_in_to_solar_buffer,
                       "PVT_OUT_FROM_SOLAR_BUFFER": T_solar_buffer_out_to_pvt,
                       "SOLAR_BUFFER_IN": T_hp_out_to_solar_buffer_in,
                       "SOLAR_BUFFER_OUT": T_hp_in_from_solar_buffer,
                       "BTES_TANK_IN": T_hp_out_to_btes_in,
                       "BTES_TANK_OUT": T_BTES_out_to_hp_in,
                       "SOLAR_HEAT_REJECTION_IN": 9999999.9,
                       "SOLAR_HEAT_REJECTION_OUT": 9999999.9,
                       "WATER_IN_EVAP": T_evap_in,
                       "WATER_OUT_EVAP": T_evap_out,
                       "WATER_IN_COND": T_cond_in,
                       "WATER_OUT_COND": T_cond_out,
                       "SH1_IN": T_hp_out_sh_in,
                       "SH1_RETURN": T_sh_out_hp_in,
                       "AIR_HP_TO_BTES_TANK": 9999999.9,
                       "DHW_INLET": T_dhw_in,
                       "DHW_OUTLET": T_dhw_out,
                       "DHW_BOTTOM": 9999999.9,
                       "SH_INLET": T_from_sh_to_demand,
                       "SH_RETURN": T_from_demand_to_space_bufer,
                       "PVT_IN": T_pvt_in,
                       "PVT_OUT": T_pvt_out,
                       "OUTDOOR_TEMP": T_air_source,
                       "BTES_TANK": T_BTES_source,
                       "SOLAR_BUFFER_TANK": T_solar_buffer_source,
                       "SH_BUFFER": T_space_heating_buffer,
                       "DHW_BUFFER": T_DHW_buffer,
                       "INDOOR_TEMP": T_indoor_temp,
                       "RECEIVER_LIQUID_IN": 9999999.9,
                       "RECEIVER_LIQUID_OUT": 9999999.9,
                       "ECO_LIQUID_OUT": 9999999.9,
                       "SUCTION_TEMP": 9999999.9,
                       "DISCHARGE_TEMP": 9999999.9,
                       "ECO_VAPOR_TEMP": 9999999.9,
                       "EXPANSION_TEMP": 9999999.9,
                       "ECO_EXPANSION_TEMP": 9999999.9,
                       "SUCTION_PRESSURE": 9999999.9,
                       "DISCHARGE_PRESSURE": 9999999.9,
                       "ECO_PRESSURE": 9999999.9,
                       "FLOW_EVAPORATOR": flow_evap,
                       "FLOW_CONDENSER": flow_condenser,
                       "FLOW_DHW": flow_dhw,
                       "FLOW_SOLAR_HEAT_REJECTION": 9999999.9,
                       "FLOW_PVT": flow_solar_circuit,
                       "FLOW_FAN_COILS_INDOOR": flow_water_demand,
                       "POWER_HP": Total_electric_power,
                       "POWER_PVT": Power_PV,
                       "PYRANOMETER": Solar_irr_tilted,
                       "Compressor_HZ": 9999999.9,
                       "EEV_LOAD1": 9999999.9,
                       "EEV_LOAD2": 9999999.9,
                       "THREE_WAY_EVAP_OPERATION": BTES_GROUND_SOLAR_VALVE,
                       "THREE_WAY_COND_OPERATION": BTES_HEATING_DHW_THREE_WAY,
                       "THREE_WAY_SOLAR_OPERATION": BTES_SOLAR_THREE_WAY_VALVE,
                       "FOUR_WAY_VALVE": HEATING_COOLING_MODE,
                       "AIR_COOLED_COMMAND": BTES_WATER_AIR_OPERATION,
                       "Residential_office_mode": Residential_office,
                       "MODBUS_LOCAL": BMES_LOCAL_CONTROL
                       }

            for key, value in new_row.items():
                if isinstance(value, float) and np.isnan(value):
                    new_row[key] = 9999999.9

            golden_df.loc[len(golden_df)] = new_row

            # Wipe Clean Test Bucket
            delete_data(url, auth_token, org)
            # Write row to Test Bucket
            write_data(new_row, url, auth_token, org)
            # Read Test Bucket
            silver_df = read_data(url, auth_token, org)
            silver_df = silver_df[golden_df.columns]

            print(f'Shape of created DataFrame is {silver_df.shape}')

            # assert not silver_df.equals(golden_df)

            if not silver_df.equals(golden_df):
                raise ValueError("Dataframes are not equal")
            else:
                print("Dataframes are equal, HOORAY")

    else:
        print("Failed to connect to PLC. Check the PLC settings and connection.")

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if client is not None and client.is_socket_open():
        client.close()
