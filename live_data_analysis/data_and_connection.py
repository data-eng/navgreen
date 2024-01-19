from pymodbus.client import ModbusTcpClient

import logging

import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
import time
import struct

from navgreen_base import establish_influxdb_connection, set_bucket, write_data, process_data


# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s')
# Set log file, its level and format
file_handler = logging.FileHandler('./logger.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
# Set stream its level and format
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def read_reg_value(register, index, divisor):
    """
    This function reads the correct value from the PLC, translates it
    to a signed integer and then converts it to the correct decimal unit
    :param register: Register to read values from
    :param index: Register's index
    :param divisor: Value's divisor to convert to preferred unit
    :return: Value converted to signed int
    """
    value = register.registers[index]
    value = struct.unpack('>h', struct.pack('>H', value))[0]
    return value / divisor

if __name__ == "__main__":

    # Establish connection with InfluxDb
    influx_client = establish_influxdb_connection()
    # Set the preferred bucket
    _ = set_bucket(os.environ.get('Bucket'))

    # PLC IP address, port and connection intervals
    plc_ip = os.environ.get('Plc_ip')
    plc_port = 502
    read_interval = 30  # Seconds
    reconnect_interval = 30  # Seconds

    # Get current date
    dataframe_date = datetime.now().strftime("%Y-%m-%d")

    client = None

    prior_checkpoint_DHW_modbus = np.nan
    prior_checkpoint_SPACE_HEATING_modbus = np.nan

    # This try-except is to catch a ctrl-c
    try:

        while True:

            try:
                # Create a Modbus TCP/IP client
                client = ModbusTcpClient(plc_ip, port=plc_port)
                client.connect()

                logger.info("Start aquiring and writing data")

                while True:
                    # You can perform read/write operations with the PLC here if needed
                    # Example: result = client.read_holding_registers(0, 1)

                    # Read holding register D500, D501, D502
                    result_coils2 = client.read_coils(8192 + 509, 7, int=0)  # read coils from 509 to 515
                    result_coils = client.read_coils(8192 + 500, 4)  # Read coils from 500 to 503

                    result = client.read_holding_registers(500, 20)  # Read registers from 500 to 519
                    results_registers2 = client.read_holding_registers(520, 18)  # Read registers from 520 to 527
                    setpoint_registers = client.read_holding_registers(540, 2)  # Read holding registers 540 and 541, setpoint SH and setpoint DHW

                    other_registers = client.read_holding_registers(542, 25)  # Read remaining pressure and temp registers

                    # Read from PLC and convert to specific unit

                    # READ COILS FROM PLC
                    Air_source_valve = result_coils.bits[0]
                    BTES_source_valve = result_coils.bits[1]
                    PVT_setting_valve = result_coils.bits[2]
                    Condenser_three_way = result_coils.bits[3]

                    # READ COILS AGAIN
                    Residential_office = result_coils2.bits[0]
                    BTES_HEATING_DHW_THREE_WAY = result_coils2.bits[1]
                    BTES_GROUND_SOLAR_VALVE = result_coils2.bits[2]
                    BTES_SOLAR_THREE_WAY_VALVE = result_coils2.bits[3]
                    BTES_WATER_AIR_OPERATION = result_coils2.bits[4]
                    HEATING_COOLING_MODE = result_coils2.bits[5]
                    BMES_LOCAL_CONTROL = result_coils2.bits[6]

                    # READ PLC REGISTERS
                    T_cond_out = read_reg_value(result, 0, 10)
                    T_cond_in = read_reg_value(result, 1, 10)
                    flow_condenser = read_reg_value(result, 2, 10000)
                    T_evap_out = read_reg_value(result, 3, 10)
                    T_evap_in = read_reg_value(result, 4, 10)
                    flow_evap = read_reg_value(result, 5, 10000)
                    T_air_source = read_reg_value(result, 6, 10)
                    T_BTES_source = read_reg_value(result, 7, 10)
                    T_solar_buffer_source = read_reg_value(result, 8, 10)
                    T_space_heating_buffer = read_reg_value(result, 9, 10)
                    T_DHW_buffer = read_reg_value(result, 10, 10)
                    T_indoor_temp = read_reg_value(result, 11, 10)
                    T_dhw_out = read_reg_value(result, 12, 10)
                    T_dhw_in = read_reg_value(result, 13, 10)
                    flow_dhw = read_reg_value(result, 14, 10000)
                    T_hp_out_sh_in = read_reg_value(result, 15, 10)
                    T_sh_out_hp_in = read_reg_value(result, 16, 10)
                    T_hp_out_to_dhw_tank = read_reg_value(result, 17, 10)
                    T_hp_in_from_dhw_tank = read_reg_value(result, 18, 10)
                    Total_electric_power = read_reg_value(result, 19, 1000)

                    # READ HOLDING REGISTERS AGAIN
                    T_from_sh_to_demand = read_reg_value(results_registers2, 0, 10)
                    T_from_demand_to_space_bufer = read_reg_value(results_registers2, 1, 10)
                    flow_water_demand = read_reg_value(results_registers2, 2, 10000)
                    Power_PV = read_reg_value(results_registers2, 3, 10000)
                    Solar_irr_tilted = read_reg_value(results_registers2, 4, 10000)
                    T_pvt_in = read_reg_value(results_registers2, 5, 10)
                    T_pvt_out = read_reg_value(results_registers2, 6, 10)
                    T_pvt_in_to_dhw = read_reg_value(results_registers2, 7, 10)
                    T_dhw_out_to_pvt = read_reg_value(results_registers2, 8, 10)
                    T_pvt_in_to_solar_buffer = read_reg_value(results_registers2, 9, 10)
                    T_solar_buffer_out_to_pvt = read_reg_value(results_registers2, 10, 10)
                    flow_solar_circuit = read_reg_value(results_registers2, 11, 10000)
                    T_hp_out_to_solar_buffer_in = read_reg_value(results_registers2, 12, 10)
                    T_hp_in_from_solar_buffer = read_reg_value(results_registers2, 13, 10)
                    T_hp_out_to_btes_in = read_reg_value(results_registers2, 14, 10)
                    T_BTES_out_to_hp_in = read_reg_value(results_registers2, 15, 10)

                    # READ REMAINING PRESSURES AND TEMP
                    T_cond_out_ref = read_reg_value(other_registers, 0, 10)
                    T_evap_out_ref = read_reg_value(other_registers, 1, 10)
                    T_cond_in_ref = read_reg_value(other_registers, 2, 10)
                    T_receiv_in_liq_ref = read_reg_value(other_registers, 3, 10)
                    T_receiv_out_liq_ref = read_reg_value(other_registers, 4, 10)
                    T_eco_liq_out_ref = read_reg_value(other_registers, 5, 10)
                    T_suction_ref = read_reg_value(other_registers, 6, 10)
                    T_discharge_ref = read_reg_value(other_registers, 7, 10)
                    T_eco_vap_ref = read_reg_value(other_registers, 8, 10)
                    T_expansion_ref = read_reg_value(other_registers, 9, 10)
                    T_eco_expansion_ref = read_reg_value(other_registers, 10, 10)

                    P_suction = read_reg_value(other_registers, 18, 10)
                    P_discharge = read_reg_value(other_registers, 19, 10)
                    P_eco = read_reg_value(other_registers, 20, 10)

                    eev_main_percentage = read_reg_value(other_registers, 21, 10)
                    eev_eco_percentage = read_reg_value(other_registers, 22, 10)

                    T_dhw_bottom = read_reg_value(other_registers, 23, 10)

                    compressor_HZ = read_reg_value(other_registers, 24, 10)

                    # READ SET-POINTS
                    T_setpoint_DHW_modbus = read_reg_value(setpoint_registers, 1, 10)
                    T_setpoint_SPACE_HEATING_modbus = read_reg_value(setpoint_registers, 0, 10)

                    # Round the set-points so that the code is not sensitive to changes beyond the second decimal
                    T_setpoint_DHW_modbus = round(T_setpoint_DHW_modbus, 2)
                    T_setpoint_SPACE_HEATING_modbus = round(T_setpoint_SPACE_HEATING_modbus, 2)

                    if T_setpoint_DHW_modbus != prior_checkpoint_DHW_modbus or T_setpoint_SPACE_HEATING_modbus != prior_checkpoint_SPACE_HEATING_modbus:
                        prior_checkpoint_DHW_modbus = T_setpoint_DHW_modbus
                        prior_checkpoint_SPACE_HEATING_modbus = T_setpoint_SPACE_HEATING_modbus
                    else:
                        T_setpoint_DHW_modbus = np.nan
                        T_setpoint_SPACE_HEATING_modbus = np.nan

                    # Get some dates
                    current_datetime = datetime.utcnow()
                    current_datetime = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

                    current_date_local = datetime.now().strftime("%Y-%m-%d")

                    if current_date_local != dataframe_date:
                        dataframe_date = current_date_local

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
                               "DHW_BOTTOM": T_dhw_bottom,
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
                               "RECEIVER_LIQUID_IN": T_receiv_in_liq_ref,
                               "RECEIVER_LIQUID_OUT": T_receiv_out_liq_ref,
                               "ECO_LIQUID_OUT": T_eco_liq_out_ref,
                               "SUCTION_TEMP": T_evap_out_ref,
                               "DISCHARGE_TEMP": T_discharge_ref,
                               "ECO_VAPOR_TEMP": T_eco_vap_ref,
                               "EXPANSION_TEMP": T_expansion_ref,
                               "ECO_EXPANSION_TEMP": T_eco_expansion_ref,
                               "SUCTION_PRESSURE": P_suction,
                               "DISCHARGE_PRESSURE": P_discharge,
                               "ECO_PRESSURE": P_eco,
                               "FLOW_EVAPORATOR": flow_evap,
                               "FLOW_CONDENSER": flow_condenser,
                               "FLOW_DHW": flow_dhw,
                               "FLOW_SOLAR_HEAT_REJECTION": np.nan,
                               "FLOW_PVT": flow_solar_circuit,
                               "FLOW_FAN_COILS_INDOOR": flow_water_demand,
                               "POWER_HP": Total_electric_power,
                               "POWER_PVT": Power_PV,
                               "PYRANOMETER": Solar_irr_tilted,
                               "Compressor_HZ": compressor_HZ,
                               "EEV_LOAD1": eev_main_percentage, "EEV_LOAD2": eev_eco_percentage,
                               "THREE_WAY_EVAP_OPERATION": BTES_GROUND_SOLAR_VALVE,
                               "THREE_WAY_COND_OPERATION": BTES_HEATING_DHW_THREE_WAY,
                               "THREE_WAY_SOLAR_OPERATION": BTES_SOLAR_THREE_WAY_VALVE,
                               "FOUR_WAY_VALVE": HEATING_COOLING_MODE,
                               "AIR_COOLED_COMMAND": BTES_WATER_AIR_OPERATION,
                               "Residential_office_mode": Residential_office,
                               "MODBUS_LOCAL": BMES_LOCAL_CONTROL,
                               "T_CHECKPOINT_DHW_MODBUS" : T_setpoint_DHW_modbus,
                                "T_CHECKPOINT_SPACE_HEATING_MODBUS": T_setpoint_SPACE_HEATING_modbus
                               }

                    # Create a new row with the current local date and time
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_row_with_time = {'Date_time_local': now, **new_row}  # Add the date and time to the new row

                    # Process columns and outliers of measurements
                    temp_df = pd.DataFrame(columns=new_row_with_time.keys())
                    temp_df.loc[0] = new_row_with_time
                    write_row = process_data(temp_df, hist_data=False)
                    write_row = write_row.iloc[0].to_dict()

                    csv_file = f'C:/Users/res4b/Desktop/modbus_tcp_ip/data/data_from_plc_{dataframe_date}.csv'

                    # If file does not exist aka the day has changed, and we need a new .csv, create it
                    # Write row to DataFrame.csv
                    if not os.path.isfile(csv_file):
                        with open(csv_file, 'w', newline='') as file:
                            writer = csv.DictWriter(file, fieldnames=new_row_with_time.keys())
                            writer.writeheader()

                    # Append the new row of information
                    with open(csv_file, 'a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=new_row_with_time.keys())
                        writer.writerow(new_row_with_time)

                    # Start the process of writing into InfluxDB
                    log_data_file_path = f'C:/Users/res4b/Desktop/modbus_tcp_ip/influx_log_data.csv'

                    # There is logged data
                    if os.path.exists(log_data_file_path):
                        # Load the data into a DataFrame
                        log_df = pd.read_csv(log_data_file_path)
                        log_df = log_df.astype(object)

                        try:
                            # Try and write logged data to DataBase
                            for idx in range(log_df.shape[0]):
                                write_data(log_df.iloc[idx], influx_client)
                            # Write also the new line,
                            write_data(write_row, influx_client)
                            # Since everything is written, delete logging dataframe
                            os.remove(log_data_file_path)
                            logger.info("Acquired connection to InfluxDB and wrote all logged data")

                        except Exception as e:
                            logger.error(f"Failed to write to InfluxDB: {str(e)}")
                            # If write was not successful, continue the logging
                            # Append the unwritten row to the file
                            with open(log_data_file_path, 'a', newline='') as file:
                                writer = csv.DictWriter(file, fieldnames=write_row.keys())
                                writer.writerow(write_row)

                            logger.info(f"Sleeping for {reconnect_interval} seconds..")
                            time.sleep(reconnect_interval)
                            continue
                    # No logged data
                    else:
                        try:
                            # Write row to DataBase
                            write_data(write_row, influx_client)

                        except Exception as e:
                            # There is an error in writing the data to InfluxDB so
                            # store it in order to write it later

                            # Create the file if it doesn't exist
                            with open(log_data_file_path, 'w', newline='') as file:
                                writer = csv.DictWriter(file, fieldnames=write_row.keys())
                                writer.writeheader()

                            # Append the new row of information
                            with open(log_data_file_path, 'a', newline='') as file:
                                writer = csv.DictWriter(file, fieldnames=write_row.keys())
                                writer.writerow(write_row)

                            logger.error(f"Failed to write to InfluxDB: {str(e)}")
                            logger.info(f"Sleeping for {reconnect_interval} seconds..")
                            time.sleep(reconnect_interval)
                            continue

                    # Wait for the specified interval
                    time.sleep(read_interval)

            # PLC exception raised
            except Exception as e:
                logger.error(f"{e}")
                logger.info(f"Sleeping for {reconnect_interval} seconds..")
                time.sleep(reconnect_interval)
                continue

    # Close connection when interrupted by the user
    except KeyboardInterrupt:
        if client is not None and client.is_socket_open():
            client.close()
            logger.info("Closed socket")