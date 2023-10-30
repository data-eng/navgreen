from pymodbus.client import ModbusTcpClient
import influxdb_client
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os

from data_and_connection import write_data
from data_and_connection import temp_sensors, pressure, solar


def create_dataframe():
    """
    Creates DataFrame with fixed columns.
    :return: The DataFrame
    """
    # Create Pandas DataFrame with the corresponding columns
    df = pd.DataFrame(columns=['DATETIME', 'PVT_IN_TO_DHW', 'PVT_OUT_FROM_DHW', 'PVT_IN_TO_SOLAR_BUFFER',
                               'PVT_OUT_FROM_SOLAR_BUFFER', 'SOLAR_BUFFER_IN', 'SOLAR_BUFFER_OUT', 'BTES_TANK_IN',
                               'BTES_TANK_OUT', 'SOLAR_HEAT_REJECTION_IN', 'SOLAR_HEAT_REJECTION_OUT',
                               '3-WAY_EVAP_OPERATION', '3-WAY_COND_OPERATION', '3-WAY_SOLAR_OPERATION',
                               'SPARE_NTC_SENSOR', 'RECEIVER_LIQUID_IN', 'RECEIVER_LIQUID_OUT', 'ECO_LIQUID_OUT',
                               'SUCTION_TEMP', 'DISCHARGE_TEMP', 'ECO_VAPOR_TEMP', 'EXPANSION_TEMP',  'ECO_PRESSURE',
                               'ECO_EXPANSION_TEMP', 'SUCTION_PRESSURE', 'DISCHARGE_PRESSURE', 'AIR_COOLED_COMMAND',
                               '4_WAY_VALVE', 'WATER_IN_EVAP', 'WATER_OUT_EVAP', 'WATER_IN_COND', 'WATER_OUT_COND',
                               'OUTDOOR_TEMP', 'BTES_TANK', 'SOLAR_BUFFER_TANK', 'SH_BUFFER', 'DHW_BUFFER',
                               'INDOOR_TEMP', 'DHW_INLET', 'DHW_OUTLET', 'SH1_IN', 'SH1_RETURN', 'DHW_BOTTOM',
                               'AIR_HP_TO_BTES_TANK', 'SH_INLET', 'SH_RETURN', 'PVT_IN', 'PVT_OUT', 'POWER_HP',
                               'POWER_GLOBAL_SOL', 'POWER_PVT', 'FLOW_EVAPORATOR', 'FLOW_CONDENSER', 'FLOW_DHW',
                               'FLOW_SOLAR_HEAT_REJECTION', 'FLOW_PVT', 'FLOW_FAN_COILS_INDOOR', 'PYRANOMETER',
                               'Compressor_HZ', 'Residential_office_mode', 'MODBUS_LOCAL', 'EEV_LOAD1', 'EEV_LOAD2'])

    df.rename(columns={
        "3-WAY_EVAP_OPERATION": "THREE_WAY_EVAP_OPERATION",
        "3-WAY_COND_OPERATION": "THREE_WAY_COND_OPERATION",
        "3-WAY_SOLAR_OPERATION": "THREE_WAY_SOLAR_OPERATION",
        "4_WAY_VALVE": "FOUR_WAY_VALVE"}, inplace=True)

    # Marked as "not important"
    df.drop("SPARE_NTC_SENSOR", axis=1, inplace=True)
    df.drop("POWER_GLOBAL_SOL", axis=1, inplace=True)

    print("Finished preprocessing, shape: {}".format(df.shape))

    return df


def create_sample_input():
    """
    Reads 100 measurements from the PLC and stores them in a DataFrame.
    :return: The DataFrame
    """

    sample_input = create_dataframe()

    # PLC IP address and port
    plc_ip = os.environ.get('Plc_ip')
    plc_port = 502

    client = None
    twiched_values = []

    for index in range(0, 100):
        try:
            # Create a Modbus TCP/IP client
            client = ModbusTcpClient(plc_ip, port=plc_port)

            # Try to establish a connection
            if client.connect():
                print("Connected to PLC successfully!")

                # Read holding register D500, D501, D502
                result_coils2 = client.read_coils(8192 + 509, 7, int=0)  # read coils 509 - 515
                result = client.read_holding_registers(500, 20)  # Read registers 500 - 519
                result_coils = client.read_coils(8192 + 500, 4)  # Read coils 500 - 503
                results_registers2 = client.read_holding_registers(520, 18)  # Read registers 520 - 527

                if result.isError():
                    print("Error reading register.")
                else:
                    t_cond_out = result.registers[0]
                    t_cond_in = result.registers[1]
                    flow_condenser = result.registers[2]
                    t_evap_out = result.registers[3]
                    t_evap_in = result.registers[4]
                    flow_evap = result.registers[5]
                    t_air_source = result.registers[6]
                    t_btes_source = result.registers[7]
                    t_solar_buffer_source = result.registers[8]
                    t_space_heating_buffer = result.registers[9]
                    t_dhw_buffer = result.registers[10]
                    t_indoor_temp = result.registers[11]
                    t_dhw_out = result.registers[12]
                    t_dhw_in = result.registers[13]
                    flow_dhw = result.registers[14]
                    t_hp_out_sh_in = result.registers[15]
                    t_sh_out_hp_in = result.registers[16]
                    # t_hp_out_to_dhw_tank = result.registers[17]
                    # t_hp_in_from_dhw_tank = result.registers[18]
                    total_electric_power = result.registers[19]
                    # air_source_valve = result_coils.bits[0]
                    # btes_source_valve = result_coils.bits[1]
                    # pvt_setting_valve = result_coils.bits[2]
                    # condenser_three_way = result_coils.bits[3]
                    t_from_sh_to_demand = results_registers2.registers[0]
                    t_from_demand_to_space_buffer = results_registers2.registers[1]
                    flow_water_demand = results_registers2.registers[2]
                    power_pv = results_registers2.registers[3]
                    solar_irr_tilted = results_registers2.registers[4]
                    t_pvt_in = results_registers2.registers[5]
                    t_pvt_out = results_registers2.registers[6]
                    t_pvt_in_to_dhw = results_registers2.registers[7]
                    t_dhw_out_to_pvt = results_registers2.registers[8]
                    t_pvt_in_to_solar_buffer = results_registers2.registers[9]
                    t_solar_buffer_out_to_pvt = results_registers2.registers[10]
                    flow_solar_circuit = results_registers2.registers[11]
                    t_hp_out_to_solar_buffer_in = results_registers2.registers[12]
                    t_hp_in_from_solar_buffer = results_registers2.registers[13]
                    t_hp_out_to_btes_in = results_registers2.registers[14]
                    t_btes_out_to_hp_in = results_registers2.registers[15]
                    # t_setpoint_DHW_modbus = results_registers2.registers[16]
                    # t_setpoint_SPACE_HEATING_modbus = results_registers2.registers[17]

                    residential_office = result_coils2.bits[0]
                    btes_heating_dhw_three_way = result_coils2.bits[1]
                    btes_ground_solar_valve = result_coils2.bits[2]
                    btes_solar_three_way_valve = result_coils2.bits[3]
                    btes_water_air_operation = result_coils2.bits[4]
                    heating_cooling_mode = result_coils2.bits[5]
                    bmes_local_control = result_coils2.bits[6]

                    current_datetime = datetime.utcnow()
                    current_datetime = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

                    new_row = {"DATETIME": current_datetime,
                               "PVT_IN_TO_DHW": t_pvt_in_to_dhw, "PVT_OUT_FROM_DHW": t_dhw_out_to_pvt,
                               "PVT_IN_TO_SOLAR_BUFFER": t_pvt_in_to_solar_buffer,
                               "PVT_OUT_FROM_SOLAR_BUFFER": t_solar_buffer_out_to_pvt,
                               "SOLAR_BUFFER_IN": t_hp_out_to_solar_buffer_in,
                               "SOLAR_BUFFER_OUT": t_hp_in_from_solar_buffer, "BTES_TANK_IN": t_hp_out_to_btes_in,
                               "BTES_TANK_OUT": t_btes_out_to_hp_in, "SOLAR_HEAT_REJECTION_IN": np.nan,
                               "SOLAR_HEAT_REJECTION_OUT": np.nan, "WATER_IN_EVAP": t_evap_in,
                               "WATER_OUT_EVAP": t_evap_out, "WATER_IN_COND": t_cond_in,
                               "WATER_OUT_COND": t_cond_out, "SH1_IN": t_hp_out_sh_in,
                               "SH1_RETURN": t_sh_out_hp_in, "AIR_HP_TO_BTES_TANK": np.nan,
                               "DHW_INLET": t_dhw_in, "DHW_OUTLET": t_dhw_out,
                               "DHW_BOTTOM": np.nan, "SH_INLET": t_from_sh_to_demand,
                               "SH_RETURN": t_from_demand_to_space_buffer, "PVT_IN": t_pvt_in,
                               "PVT_OUT": t_pvt_out, "OUTDOOR_TEMP": t_air_source, "BTES_TANK": t_btes_source,
                               "SOLAR_BUFFER_TANK": t_solar_buffer_source, "SH_BUFFER": t_space_heating_buffer,
                               "DHW_BUFFER": t_dhw_buffer, "INDOOR_TEMP": t_indoor_temp,
                               "RECEIVER_LIQUID_IN": np.nan, "RECEIVER_LIQUID_OUT": np.nan, "ECO_LIQUID_OUT": np.nan,
                               "SUCTION_TEMP": np.nan, "DISCHARGE_TEMP": np.nan, "ECO_VAPOR_TEMP": np.nan,
                               "EXPANSION_TEMP": np.nan, "ECO_EXPANSION_TEMP": np.nan, "SUCTION_PRESSURE": np.nan,
                               "DISCHARGE_PRESSURE": np.nan, "ECO_PRESSURE": np.nan, "FLOW_EVAPORATOR": flow_evap,
                               "FLOW_CONDENSER": flow_condenser, "FLOW_DHW": flow_dhw,
                               "FLOW_SOLAR_HEAT_REJECTION": np.nan, "FLOW_PVT": flow_solar_circuit,
                               "FLOW_FAN_COILS_INDOOR": flow_water_demand, "POWER_HP": total_electric_power,
                               "POWER_PVT": power_pv, "PYRANOMETER": solar_irr_tilted, "Compressor_HZ": np.nan,
                               "EEV_LOAD1": np.nan, "EEV_LOAD2": np.nan,
                               "THREE_WAY_EVAP_OPERATION": btes_ground_solar_valve,
                               "THREE_WAY_COND_OPERATION": btes_heating_dhw_three_way,
                               "THREE_WAY_SOLAR_OPERATION": btes_solar_three_way_valve,
                               "FOUR_WAY_VALVE": heating_cooling_mode, "AIR_COOLED_COMMAND": btes_water_air_operation,
                               "Residential_office_mode": residential_office, "MODBUS_LOCAL": bmes_local_control}

                    # Randomly add some values that are outliers
                    random_int = random.randint(0, 2)

                    if random_int == 0:  # Change Temperatures
                        i = random.randint(0, len(temp_sensors) - 1)
                        row[temp_sensors[i]] = -21.0 if random.randint(0, 1) == 0 else 101.0

                        twiched_values += [(index, temp_sensors[i])]

                    elif random_int == 0:  # Change Pressures
                        i = random.randint(0, len(pressure) - 1)
                        row[pressure[i]] = -1.0 if random.randint(0, 1) == 0 else 36.0

                        twiched_values += [(index, pressure[i])]

                    else:  # Change Solar
                        i = random.randint(0, len(solar) - 1)
                        row[solar[i]] = 7.0

                        twiched_values += [(index, solar[i])]

                    # Add new row to the sample input dataframe
                    sample_input[len(sample_input)] = new_row
                    sample_input.to_csv("./test/sample_input.csv", index=False)
            else:
                print("Failed to connect to PLC. Check the PLC settings and connection.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

        finally:
            if client is not None and client.is_socket_open():
                client.close()

    return sample_input


def delete_data(url, token, organization, bucket):
    """
    Deletes all data from a specified bucket.
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: None
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.delete_api()
    api.delete(bucket=f'"{bucket}"', org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="*"')

    print("Deleted ok")


def read_data(url, token, organization, bucket):
    """
    Reads data from a specified bucket and stores it in a DataFrame
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: The DataFrame
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.query_api()
    query = f'from(bucket: "{bucket}") |> range(start: 0)'
    data = api.query(org=organization, query=query)
    print("Read ok")

    # Extract the records from the result
    records = []
    for table in data:
        for record in table.records:
            print(f'{record["_field"]} || {record["_value"]}')
            records.append(record.values)

    # Create a list to store the unique "field" values
    unique_fields = list(set(record["_field"] for record in records))
    print(len(unique_fields))
    d = dict()

    # Add a column for each unique "field" value
    for name in unique_fields:
        name_data = [record["_value"] for record in records if record["_field"] == name]
        d[name] = name_data

    sample_output = pd.DataFrame(d)
    sample_output.to_csv("./test/sample_output.csv", index=False)

    return sample_output


if __name__ == "__main__":

    # Create 'sample input' from live raw data OR load the existing
    while True:
        user_input = input("Do you want to regenerate the sample input? Please enter 'y' or 'n': ").lower()

        if user_input == 'y':
            print("You chose 'y'.")
            break
        elif user_input == 'n':
            print("You chose 'n'.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    sample_input = create_sample_input() if user_input == 'y' else pd.read_csv("./test/sample_input.csv")

    # Import credentials
    url = os.environ.get('Url_influx_db')
    org = os.environ.get('Organization_influx')
    auth_token = os.environ.get('Auth_token')
    bucket = os.environ.get('Bucket')

    # Wipe clean the test bucket
    delete_data(url, auth_token, org, bucket)
    
    # Read each data sample and write it to the test bucket
    # Function 'write_data' is imported from the ingestion script and does the preprocessing etc
    for _, row in sample_input.iterrows():
        write_data(row, url, auth_token, org, bucket)

    # Get the 'sample output' by querying the test bucket
    sample_output = read_data(url, auth_token, org, bucket)
