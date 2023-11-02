from pymodbus.client import ModbusTcpClient

import pandas as pd
import numpy as np

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import time

import os


def make_point(measurement, row, value_columns, tag_columns):
    p = influxdb_client.Point(measurement)
    p.time(row["DATETIME"])

    # Tag with the state of the valves, as context
    for col in tag_columns:
        if row[col] is not np.nan:
            p.tag(col, row[col])
    # Add the sensor data fields
    for col in value_columns:
        if row[col] is not np.nan:
            p.field(col, row[col])
    return p


def write_data(row, url, token, organization, bucket):
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.write_api(write_options=SYNCHRONOUS)

    # Apply reasonable limits
    for col in solar:
        if row[col] > 2.0:  # Error that happens at nighttime
            row[col] = 0.0
    for col in temp_sensors:
        if row[col] < -20.0 or row[col] > 100.0:
            row[col] = np.nan
    for col in pressure:
        if row[col] < 0.0 or row[col] > 35.0:
            row[col] = np.nan

    p = make_point("pressure", row, pressure, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("temperature", row, temp_sensors, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("flow", row, flow, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("power", row, power, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("solar", row, solar, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("other", row, other, control)
    api.write(bucket=bucket, org=organization, record=p)
    # Also add controls as values, for viz
    p = make_point("control", row, control, [])
    api.write(bucket=bucket, org=organization, record=p)
    # print("Write ok")


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

temp_sensors = []
temp_sensors.extend(water_temp)
temp_sensors.extend(other_temp)
temp_sensors.extend(ref_temp)


if __name__ == "__main__":

    # Create Pandas DataFrame with the corresponding columns
    df = pd.DataFrame(
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
                 'MODBUS_LOCAL', 'EEV_LOAD1', 'EEV_LOAD2'])

    df.rename(columns={
        "3-WAY_EVAP_OPERATION": "THREE_WAY_EVAP_OPERATION",
        "3-WAY_COND_OPERATION": "THREE_WAY_COND_OPERATION",
        "3-WAY_SOLAR_OPERATION": "THREE_WAY_SOLAR_OPERATION",
        "4_WAY_VALVE": "FOUR_WAY_VALVE"}, inplace=True)

    print("Finished loading CSV, shape: {}".format(df.shape))

    # Marked as "not important"
    df.drop("SPARE_NTC_SENSOR", axis=1, inplace=True)
    df.drop("POWER_GLOBAL_SOL", axis=1, inplace=True)

    # Redundant
    df.drop("Date", axis=1, inplace=True)
    df.drop("Time", axis=1, inplace=True)

    print("Finished preprocessing, shape: {}".format(df.shape))

    # just checking that nothing is forgotten. +1 for datetime
    assert ((len(df.columns) - 1) == len(water_temp) + len(other_temp) + len(pressure) + len(flow) + len(power)
            + len(solar) + len(other) + len(control) + len(ref_temp) + 1)

    # CREDENTIALS: CAREFUL
    url = os.environ.get('Url_influx_db')
    tok = os.environ.get('Write_token')
    org = os.environ.get('Organization_influx')
    bucket = os.environ.get('Bucket')

    # PLC IP address, port and connection intervals
    plc_ip = os.environ.get('Plc_ip')
    plc_port = 502
    read_interval = 30  # Seconds
    reconnect_interval = 10  # Seconds

    # Get current date
    dataframe_date = datetime.now().strftime("%Y-%m-%d")

    while True:

        client = None
        try:
            # Create a Modbus TCP/IP client
            client = ModbusTcpClient(plc_ip, port=plc_port)

            # Try to establish a connection
            if client.connect():
                print("Connected to PLC successfully!")
                # You can perform read/write operations with the PLC here if needed
                # Example: result = client.read_holding_registers(0, 1)

                while True:
                    # Read holding register D500, D501, D502
                    result_coils2 = client.read_coils(8192 + 509, 7, int=0)  # read coils from 509 to 515
                    result = client.read_holding_registers(500, 20)  # Read registers from 500 to 519
                    result_coils = client.read_coils(8192 + 500, 4)  # Read coils from 500 to 503
                    results_registers2 = client.read_holding_registers(520, 18)  # Read registers from 520 to 527

                    if result.isError():
                        print("Error reading register.")
                    else:

                        # Read from PLC and convert to specific unit

                        # READ PLC REGISTERS
                        T_cond_out = result.registers[0] / 10
                        # print(f"T_condenser_out: {T_cond_out}")
                        T_cond_in = result.registers[1] / 10
                        # print(f"T_condenser_in: {T_cond_in}")
                        flow_condenser = result.registers[2] / 10000  # convert to m^3/h
                        # print(f"flow_condenser: {flow_condenser}")
                        T_evap_out = result.registers[3] / 10
                        # print(f"T_evap_out: {T_evap_out}")
                        T_evap_in = result.registers[4] / 10
                        # print(f"T_evap_in: {T_evap_in}")
                        flow_evap = result.registers[5] / 10000  # convert to m^3/h
                        # print(f"flow_evaporator: {flow_evap}")
                        # Read_Ambient air temperature
                        T_air_source = result.registers[6] / 10
                        # print(f"T_air_source: {T_air_source}")
                        T_BTES_source = result.registers[7] / 10
                        # print(f"T_btes_source: {T_BTES_source}")
                        # Read_solar_buffer_TEMPERATURE
                        T_solar_buffer_source = result.registers[8] / 10
                        # print(f"T_solar_buffer_source: {T_solar_buffer_source}")
                        # Read_space_heating_temperature
                        T_space_heating_buffer = result.registers[9] / 10
                        # print(f"T_space_heating_buffer: {T_space_heating_buffer}")
                        # Read_solar_buffer_TEMPERATURE
                        T_DHW_buffer = result.registers[10] / 10
                        # print(f"T_DHW_buffer: {T_DHW_buffer}")
                        # Read_indoor_TEMPERATURE
                        T_indoor_temp = result.registers[11] / 10
                        # print(f"T_indoor_temperature: {T_indoor_temp}")
                        # Read_DHW_outlet_temperature
                        T_dhw_out = result.registers[12] / 10
                        # print(f"T_dhw_outlet_temp: {T_dhw_out}")
                        # Read_DHW_inlet_TEMPERATURE
                        T_dhw_in = result.registers[13] / 10
                        # print(f"T_dhw_inlet_temp: {T_dhw_in}")
                        # Read_indoor_TEMPERATURE
                        flow_dhw = result.registers[14] / 10000  # convert to m^3/h
                        # print(f"flow_dhw: {flow_dhw}")
                        # Read_sh_inlet_TEMPERATURE_from_heat_pump
                        T_hp_out_sh_in = result.registers[15] / 10
                        # print(f"T_sh_in_from_HP: {T_hp_out_sh_in}")
                        # Read_sh_outlet_TEMPERATURE_to_heat_pump
                        T_sh_out_hp_in = result.registers[16] / 10
                        # print(f"T_sh_in_from_HP: {T_hp_out_sh_in}")
                        # Read_hp_outlet_to_DHW_tank_(same_as_condenser_out)
                        T_hp_out_to_dhw_tank = result.registers[17] / 10
                        # print(f"T_hp_out_to_dhw_tank: {T_hp_out_to_dhw_tank}")
                        # Read_hp_in_temperature_from_DHW_tank(same_as_condenser_in)
                        T_hp_in_from_dhw_tank = result.registers[18] / 10
                        # print(f"T_hp_in_from_dhw_tank: {T_hp_in_from_dhw_tank}")
                        # Read_hp_in_temperature_from_DHW_tank(same_as_condenser_in)
                        # Electricity in kW"
                        Total_electric_power = result.registers[19] / 1000
                        # print(f"POWER_HP: {Total_electric_power}")

                        # READ COILS FROM PLC
                        # Read_air_source_command
                        Air_source_valve = result_coils.bits[0]
                        # IF OFF WATER SOURCE, IF ON AIR SOURCE
                        # print(f"Air_source_command: {Air_source_valve}")
                        # Read_BTES_SOURCE_valve_setting
                        # if OFF solar buffer, if ON BTES tank
                        BTES_source_valve = result_coils.bits[1]
                        # print(f"BTES_source_command: {BTES_source_valve}")
                        # Read_PVT_flow_valve_setting
                        # IF ON THEN DHW TANK, if OFF solar buffer tank
                        PVT_setting_valve = result_coils.bits[2]
                        # print(f"PVT_setting_valve: {PVT_setting_valve}")
                        # Condenser_heat_output_direction
                        # IF ON HEAT TO SPACE BUFFER, IF OFF HEAT TO DHW TANK
                        Condenser_three_way = result_coils.bits[3]
                        # print(f"Condenser_three_way: {Condenser_three_way}")

                        # READ HOLDING REGISTERS AGAIN
                        # Read_temperature_from_space_buffer_to_demand
                        T_from_sh_to_demand = results_registers2.registers[0] / 10
                        # print(f"T_from_sh_to_demand: {T_from_sh_to_demand}")
                        # Read_temperature_from_demand_to_space_buffer
                        T_from_demand_to_space_bufer = results_registers2.registers[1] / 10
                        # print(f"T_from_demand_to_space_bufer: {T_from_demand_to_space_bufer}")
                        # water_flow_rate_for_space_heating_or_cooling
                        flow_water_demand = results_registers2.registers[2] / 10000  # convert to m^3/h
                        # print(f"flow_water_demand: {flow_water_demand}")
                        # Read_electrical_power_produced_by_PVT_collectors
                        Power_PV = results_registers2.registers[3] / 10000  # CONVERT TO kW
                        # print(f"Power_PV: {Power_PV}")
                        # Read_tilted_collector_solar_irradiation
                        # Units: kW / m^2
                        Solar_irr_tilted = results_registers2.registers[4] / 10000  # CONVERT TO kW / m^2
                        print(f"Solar_irr_tilted: {Solar_irr_tilted}")
                        # Read_temperature_PVT_inlet
                        T_pvt_in = results_registers2.registers[5] / 10
                        # print(f"T_pvt_in: {T_pvt_in}")
                        # Read_temperature_PVT_outlet
                        T_pvt_out = results_registers2.registers[6] / 10
                        # print(f"T_pvt_out: {T_pvt_out}")
                        # Read_temperature_from_PVT____inlet_to_dhw
                        T_pvt_in_to_dhw = results_registers2.registers[7] / 10
                        # print(f"T_pvt_in_to_dhw: {T_pvt_in_to_dhw}")
                        # Read_temperature_PVT_outlet_from_dhw_back_to_collectors
                        T_dhw_out_to_pvt = results_registers2.registers[8] / 10
                        # print(f"T_dhw_out_to_pvt: {T_dhw_out_to_pvt}")
                        # Read_temperature_from_PVT____inlet_to_solar_buffer_tank
                        T_pvt_in_to_solar_buffer = results_registers2.registers[9] / 10
                        # print(f"T_pvt_in_to_solar_buffer: {T_pvt_in_to_solar_buffer}")
                        # Read_temperature_PVT_outlet_from_solar_buffer_back_to_collectors
                        T_solar_buffer_out_to_pvt = results_registers2.registers[10] / 10
                        # print(f"T_solar_buffer_out_to_pvt: {T_solar_buffer_out_to_pvt}")
                        # solar_circuit_flow_rate
                        flow_solar_circuit = results_registers2.registers[11] / 10000  # convert to m^3/h
                        # print(f"flow_solar_circuit: {flow_solar_circuit}")
                        # Read_temperature_hp_evap_out_to_solar_buffer_in
                        T_hp_out_to_solar_buffer_in = results_registers2.registers[12] / 10
                        # print(f"T_hp_out_to_solar_buffer_in: {T_hp_out_to_solar_buffer_in}")
                        # Read_temperature_hp_evap_in_from_solar_buffer_out
                        T_hp_in_from_solar_buffer = results_registers2.registers[13] / 10
                        # print(f"T_hp_in_from_solar_buffer: {T_hp_in_from_solar_buffer}")
                        # Read_temperature_hp_evap_out_to_BTES_tank_IN
                        T_hp_out_to_btes_in = results_registers2.registers[14] / 10
                        # print(f"T_hp_out_to_btes_in: {T_hp_out_to_btes_in}")
                        # Read_temperature_BTES_out_to_HP_in
                        T_BTES_out_to_hp_in = results_registers2.registers[15] / 10
                        # print(f"T_BTES_out_to_hp_in: {T_BTES_out_to_hp_in}")
                        # DHW_TANK_TEMPERATURE_SETPOINT_MODBUS_OPERATION
                        T_setpoint_DHW_modbus = results_registers2.registers[16] / 10
                        # print(f"T_setpoint_DHW_modbus: {T_setpoint_DHW_modbus}")
                        # SPACE_HEATING_TANK_TEMPERATURE_SETPOINT_MODBUS_OPERATION
                        T_setpoint_SPACE_HEATING_modbus = results_registers2.registers[17] / 10
                        # print(f"T_setpoint_SPACE_HEATING_modbus: {T_setpoint_SPACE_HEATING_modbus}")

                        # READ COILS AGAIN
                        # Residential_or_office_mode
                        Residential_office = result_coils2.bits[0]
                        # IF ON MODE IS OFFICE, IF OFF MODE IS RESIDENTIAL
                        # print(f"Residential_office: {Residential_office}")

                        # COILS FOR BMES to know the valve position
                        BTES_HEATING_DHW_THREE_WAY = result_coils2.bits[1]
                        # IF BTES_HEATING_DHW_THREE_WAY = 0N THEN SPACE BUFFER, IF ITS OFF THEN DHW
                        # print(f"BTES_HEATING_DHW_THREE_WAY: {BTES_HEATING_DHW_THREE_WAY}")
                        BTES_GROUND_SOLAR_VALVE = result_coils2.bits[2]
                        # IF ON MODE IS GROUND , IF OFF MODE IS SOLAR_BUFFER
                        # print(f"BTES_GROUND_SOLAR_VALVE: {BTES_GROUND_SOLAR_VALVE}")
                        BTES_SOLAR_THREE_WAY_VALVE = result_coils2.bits[3]
                        # IF ON MODE IS DHW TANK , IF OFF MODE IS SOLAR_BUFFER
                        # print(f"BTES_SOLAR_THREE_WAY_VALVE: {BTES_SOLAR_THREE_WAY_VALVE}")
                        BTES_WATER_AIR_OPERATION = result_coils2.bits[4]
                        # IF ON MODE IS AIR SOURCE , IF OFF MODE IS WATER_SOURCE
                        # print(f"BTES_WATER_AIR_OPERATION: {BTES_WATER_AIR_OPERATION}")

                        # HEATING_COOLING_MODE
                        HEATING_COOLING_MODE = result_coils2.bits[5]
                        # IF ON MODE IS HEATING , IF OFF MODE IS COOLING
                        print(f"HEATING_COOLING_MODE: {HEATING_COOLING_MODE}")
                        # BMES_LOCAL_CONTROL
                        BMES_LOCAL_CONTROL = result_coils2.bits[6]
                        # IF ON MODE IS LOCAL , IF OFF MODE IS MODBUS
                        print(f"BMES_LOCAL_CONTROL: {BMES_LOCAL_CONTROL}")

                        # Get some dates
                        current_datetime = datetime.utcnow()
                        current_datetime = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

                        current_date_local = datetime.now().strftime("%Y-%m-%d")

                        if current_date_local != dataframe_date:
                            dataframe_date = current_date_local
                            df.drop(df.index, inplace=True)

                        # The data till now (CONVERTED) are taken into account from the TESTCASE in test_live_data.py
                        new_row = {"DATETIME": current_datetime,  # "DATETIME": pd.to_datetime(current_datetime),
                                   "PVT_IN_TO_DHW": T_pvt_in_to_dhw,
                                   "PVT_OUT_FROM_DHW": T_dhw_out_to_pvt,
                                   "PVT_IN_TO_SOLAR_BUFFER": T_pvt_in_to_solar_buffer,
                                   "PVT_OUT_FROM_SOLAR_BUFFER": T_solar_buffer_out_to_pvt,
                                   "SOLAR_BUFFER_IN": T_hp_out_to_solar_buffer_in,
                                   "SOLAR_BUFFER_OUT": T_hp_in_from_solar_buffer,
                                   "BTES_TANK_IN": T_hp_out_to_btes_in,
                                   "BTES_TANK_OUT": T_BTES_out_to_hp_in,
                                   "SOLAR_HEAT_REJECTION_IN": np.nan,
                                   "SOLAR_HEAT_REJECTION_OUT": np.nan,
                                   "WATER_IN_EVAP": T_evap_in,
                                   "WATER_OUT_EVAP": T_evap_out,
                                   "WATER_IN_COND": T_cond_in,
                                   "WATER_OUT_COND": T_cond_out,
                                   "SH1_IN": T_hp_out_sh_in,
                                   "SH1_RETURN": T_sh_out_hp_in,
                                   "AIR_HP_TO_BTES_TANK": np.nan,
                                   "DHW_INLET": T_dhw_in,
                                   "DHW_OUTLET": T_dhw_out,
                                   "DHW_BOTTOM": np.nan,
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
                                   "RECEIVER_LIQUID_IN": np.nan,
                                   "RECEIVER_LIQUID_OUT": np.nan,
                                   "ECO_LIQUID_OUT": np.nan,
                                   "SUCTION_TEMP": np.nan,
                                   "DISCHARGE_TEMP": np.nan,
                                   "ECO_VAPOR_TEMP": np.nan,
                                   "EXPANSION_TEMP": np.nan,
                                   "ECO_EXPANSION_TEMP": np.nan,
                                   "SUCTION_PRESSURE": np.nan,
                                   "DISCHARGE_PRESSURE": np.nan,
                                   "ECO_PRESSURE": np.nan,
                                   "FLOW_EVAPORATOR": flow_evap,
                                   "FLOW_CONDENSER": flow_condenser,
                                   "FLOW_DHW": flow_dhw,
                                   "FLOW_SOLAR_HEAT_REJECTION": np.nan,
                                   "FLOW_PVT": flow_solar_circuit,
                                   "FLOW_FAN_COILS_INDOOR": flow_water_demand,
                                   "POWER_HP": Total_electric_power,
                                   "POWER_PVT": Power_PV,
                                   "PYRANOMETER": Solar_irr_tilted,
                                   "Compressor_HZ": np.nan,
                                   "EEV_LOAD1": np.nan, "EEV_LOAD2": np.nan,
                                   "THREE_WAY_EVAP_OPERATION": BTES_GROUND_SOLAR_VALVE,
                                   "THREE_WAY_COND_OPERATION": BTES_HEATING_DHW_THREE_WAY,
                                   "THREE_WAY_SOLAR_OPERATION": BTES_SOLAR_THREE_WAY_VALVE,
                                   "FOUR_WAY_VALVE": HEATING_COOLING_MODE,
                                   "AIR_COOLED_COMMAND": BTES_WATER_AIR_OPERATION,
                                   "Residential_office_mode": Residential_office,
                                   "MODBUS_LOCAL": BMES_LOCAL_CONTROL
                                   }

                        # Write row to DataBase
                        write_data(new_row, url, tok, org, "mult_source_heatpump")

                        # Write row to DataFrame (future .csv)
                        # Create a new row with the current local date and time
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_row_with_time = {'Date_time_local': now, **new_row}  # Add the date and time to the new row

                        df.loc[len(df)] = new_row_with_time
                        df.to_csv(f'C:/Users/res4b/Desktop/modbus_tcp_ip/data/data_from_plc_{dataframe_date}.csv',
                                  mode='w', index=False)

                    # Wait for the specified interval
                    time.sleep(read_interval)
            else:
                print("Failed to connect to PLC. Check the PLC settings and connection.")
                time.sleep(reconnect_interval)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(reconnect_interval)
            continue

        finally:
            if client is not None and client.is_socket_open():
                client.close()
