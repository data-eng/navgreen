import time
import os
import struct
import pandas as pd
import csv
import numpy as np
from datetime import datetime, timedelta
from pymodbus.client import ModbusTcpClient

from navgreen_base import delete_data_weather, read_data_weather, set_bucket_weather, \
    establish_influxdb_connection_weather

import logging

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s')
# Set log file, its level and format
file_handler = logging.FileHandler('./remote_control_logger_weather.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
# Set stream its level and format
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_DHW(h, m):
    """
    This function returns the observed DHW for that specific time and month
    :param h: hour
    :param m: month
    :return: DWH (kWh)
    """

    file_path = './DHW_profile_KWH.csv'
    df = pd.read_csv(file_path)

    month_map = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec'
    }

    df['hour'] = [0, 3, 6, 9, 12, 15, 18, 21]

    hour = df[df['hour'] == h]

    DHW_val = hour[month_map[m]].values

    assert len(DHW_val) == 1

    return DHW_val[0]


def get_predictions():
    """
    This function reads the daily QPVT predictions from the
    InfluxDB and asserts every necessary checks.
    """

    # InfluxDB requirements
    influx_client = establish_influxdb_connection_weather()
    bucket = set_bucket_weather("model_predictions")

    # Get the predicted Q_PVT by querying the test bucket
    df = None
    while True:
        try:
            logger.info(f"Reading from {bucket}... (last 2 days)")
            df = read_data_weather(influx_client, start="-2d")
            logger.info("DONE.")
            break
        except Exception as e:
            logger.info("Reading from InfluxDB failed")
            logger.info(e)
            logger.info(f"Sleeping for {reconnect_interval} seconds and retrying...")
            time.sleep(reconnect_interval)
            continue

    assert isinstance(df, pd.DataFrame), "Object is not a pandas DataFrame"

    # Get the last 8 rows (8 x 3 hrs)
    df_predictions_today = df.tail(8).reset_index()

    # Check if the values were actually accessed the same day as this script is run
    if not (any(df_predictions_today['DATETIME'].dt.date == datetime.now().date()) and any(
            df_predictions_today['DATETIME'].dt.date == (datetime.now() + timedelta(days=1)).date())):
        raise ValueError("Today's weather predictions do not exist.")

    '''
    # Wipe clean the test bucket
    logger.info(f"Emptying {bucket}... ")
    delete_data_weather(influx_client)
    logger.info("DONE.")
    '''

    df_predictions_today.drop(columns=['index'], inplace=True)

    return df_predictions_today


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


def write_reg_value(client, register, value, multiplier):
    """
    Change the value of some registers.
    :param client: ModBus client
    :param register: Register to change
    :param value: New value (semantic)
    :param multiplier: Multiplier to make semantic value integer
    :return: None
    """
    # Convert float to integer
    new_value = int(value * multiplier)
    client.write_register(register, new_value)

    return


qpvt_class_to_range = {'0': [0.0, 0.05],
                       '1': [0.05, 0.21],
                       '2': [0.21, 0.53],
                       '3': [0.53, 1.05],
                       '4': [1.05, float('inf')]
                       }

if __name__ == "__main__":

    dataframe_date = datetime.now().strftime("%Y-%m-%d")

    # Current hour in installation
    current_hour, current_day, current_month = datetime.now().hour, datetime.now().day, datetime.now().month

    # current_hour = 15

    if current_hour in [0, 3, 6, 9, 12, 15, 18, 21]:
        row = {'DATETIME': pd.to_datetime(datetime.now()).floor('H'), 'SETPOINT_ML': 0}
    else:
        raise ValueError('Woke up at the wrong time.')

    # PLC IP address, port and connection intervals
    plc_ip = os.environ.get('Plc_ip')
    plc_port = 502
    reconnect_interval = 30  # Seconds
    PLC_connect_interval = 5 * 60

    too_cold = False
    min_setpoint, max_setpoint = 44.5, 48.0

    try:
        while True:
            # Try to connect with the InfluxDB
            try:
                # Read QPVT from server and do something with it
                predictions = get_predictions()

                qpvt_predicted = predictions[(predictions['DATETIME'].dt.hour == current_hour) &
                                             (predictions['DATETIME'].dt.day == current_day) &
                                             (predictions['DATETIME'].dt.month == current_month)]

                qpvt_predicted = qpvt_predicted['predicted']
                assert len(qpvt_predicted) == 1, f"Expected 1 row, but found {len(qpvt_predicted)} rows"

                qpvt_predicted = qpvt_predicted.iloc[0]

                dhw = get_DHW(h=current_hour, m=current_month)

                # should check when to make it run
                # also make the correct checks so that it does not loop for ever
                # check hour indexes
                # store if setpoint was from algo

                q_pvt_predicted = np.mean(qpvt_class_to_range[str(qpvt_predicted)]) if qpvt_predicted < 4 else \
                qpvt_class_to_range[str(qpvt_predicted)][0]

                if dhw < 0.0:
                    setpoint_DHW = min_setpoint
                else:
                    if qpvt_predicted >= dhw:
                        setpoint_DHW = min_setpoint
                    else:
                        setpoint_DHW = max_setpoint

                # Now, write the new setpoint to the PLC
                # Connect with the PLC
                modbus_client = None

                break

                '''try:
                    while True:
                        try:
                            # Create a Modbus TCP/IP modbus_client
                            modbus_client = ModbusTcpClient(plc_ip, port=plc_port)
                            modbus_client.connect()

                            logger.debug("Acquire and write data.")

                            # Read holding registers
                            result = modbus_client.read_holding_registers(500, 20)
                            other_registers = modbus_client.read_holding_registers(542, 25)  # Read pressure and temp registers
                            result_alarm = modbus_client.read_coils(8192 + 506, 1, int=0)
                            # setpoint_registers_write_heating = modbus_client.read_holding_registers(537, 1)

                            # Get the values we want
                            general_alarm = result_alarm.bits[0]
                            BTES_TANK = read_reg_value(result, 7, 10)
                            DHW_buffer = read_reg_value(result, 10, 10)
                            POWER_HP = read_reg_value(result, 19, 1000)
                            compressor_HZ = read_reg_value(other_registers, 24, 10)
                            # setpoint_write_heating = read_reg_value(setpoint_registers_write_heating, 0, 10)

                            # Setpoint value given by the server
                            new_DHW_setpoint = setpoint_DHW

                            write_reg_value(modbus_client, 536, new_DHW_setpoint, 10)

                            break

                            # Get ready to write:

                            # If no alarm is raised
                            if not general_alarm:
                                # If the heatpump was not turned off previously due to cold temperature
                                if not too_cold:
                                    write_value = False

                                    # Tank is too cold
                                    if BTES_TANK <= 8.0:
                                        logger.debug('BTES_TANK too cold.. closing')
                                        too_cold = True
                                        new_DHW_setpoint = min_setpoint
                                        write_value = True

                                    # If heat pump is on
                                    if compressor_HZ >= 30.0 and POWER_HP > 2.0:
                                        # Top of tank is too hot
                                        if DHW_buffer > 52.0:
                                            logger.debug('DHW_buffer too hot.. closing')
                                            # Turn off HP
                                            new_DHW_setpoint = min_setpoint

                                        write_value = True
                                    else:
                                        # If the hp is going to be turned on check that it respects the constraints
                                        if new_DHW_setpoint > DHW_buffer + 1:  # Start hp (+1 dt)
                                            if DHW_buffer <= 52.0 and BTES_TANK > 8.0:
                                                write_value = True
                                        else: # otherwise, if the hp will remain closed, set the minimal setpoint
                                            new_DHW_setpoint = min_setpoint
                                            write_value = True

                                    # Everything written if constraints allow it
                                    if write_value:
                                        write_reg_value(modbus_client, 536, new_DHW_setpoint, 10)
                                        logger.info(f'Writing setpoint with value: {new_DHW_setpoint}')

                                        # wait a bit until the PLC acquires the value
                                        time.sleep(2)

                                        try:
                                            # setpoint_registers_read_heating = modbus_client.read_holding_registers(540, 1)
                                            setpoint_registers_read_DHW = modbus_client.read_holding_registers(541, 1)
                                            setpoint_read_DHW = read_reg_value(setpoint_registers_read_DHW, 0, 10)

                                            # If we read the setpoint that the PLC reads, and it
                                            # has our value,  the operation was successful
                                            if setpoint_read_DHW == new_DHW_setpoint:
                                                logger.info("The setpoint change was successful.")
                                                row['SETPOINT_ML'] = 1
                                            else:
                                                setpoint_registers_write_DHW = modbus_client.read_holding_registers(536,1)
                                                setpoint_write_DHW = read_reg_value(setpoint_registers_write_DHW, 0, 10)

                                                # Otherwise, if the local setpoint has the value and not the PLC one,
                                                # we forgot to turn on the 'modbus' command in the PLC, which should be
                                                # done at the installation
                                                if setpoint_write_DHW == new_DHW_setpoint:
                                                    logger.info("Modbus control deactivated. Activate physically from screen.")
                                                else: # if none of it holds true, the setpoint value did not change
                                                    logger.info("Setpoint writing unsuccessful. Try again.")
                                        except:
                                            # In this case for some reason we could not read from the PLC, so we do not
                                            # know what happened with our setpoint
                                            logger.debug("Reading the PLC: unsuccessful.")
                                    else:
                                        logger.info("No writing.")

                                else:
                                    # If hp was closed due to cold temperature, should wait until it is at least 12
                                    if BTES_TANK >= 12.0:
                                        too_cold = False

                                        write_reg_value(modbus_client, 536, new_DHW_setpoint, 10)
                                        logger.info(f'Writing setpoint with value: {new_DHW_setpoint}')

                                        # wait a bit until the PLC acquires the value
                                        time.sleep(2)

                                        try:
                                            # setpoint_registers_read_heating = modbus_client.read_holding_registers(540, 1)
                                            setpoint_registers_read_DHW = modbus_client.read_holding_registers(541, 1)
                                            setpoint_read_DHW = read_reg_value(setpoint_registers_read_DHW, 0, 10)

                                            # If we read the setpoint that the PLC reads, and it
                                            # has our value,  the operation was successful
                                            if setpoint_read_DHW == new_DHW_setpoint:
                                                logger.info("The setpoint change was successful.")
                                                row['SETPOINT_ML'] = 1
                                            else:
                                                setpoint_registers_write_DHW = modbus_client.read_holding_registers(536, 1)
                                                setpoint_write_DHW = read_reg_value(setpoint_registers_write_DHW, 0, 10)

                                                # Otherwise, if the local setpoint has the value and not the PLC one,
                                                # we forgot to turn on the 'modbus' command in the PLC, which should be
                                                # done at the installation
                                                if setpoint_write_DHW == new_DHW_setpoint:
                                                    logger.info("Modbus control deactivated. Activate physically from screen.")
                                                else:  # if none of it holds true, the setpoint value did not change
                                                    logger.info("Setpoint writing unsuccessful. Try again.")
                                        except:
                                            # In this case for some reason we could not read from the PLC, so we do not
                                            # know what happened with our setpoint
                                            logger.debug("Reading the PLC: unsuccessful.")
                                    else:
                                        logger.debug('BTES_TANK still too cold.')
                            else:
                                logger.debug('Alarm is on!')

                            break

                        # PLC exception raised
                        except Exception as e:
                            logger.error(f"{e}")
                            logger.debug(f"Sleeping for {reconnect_interval} seconds..")

                            if modbus_client is not None and modbus_client.is_socket_open():
                                modbus_client.close()

                            time.sleep(reconnect_interval)
                            continue

                # Close connection when interrupted by the user
                except KeyboardInterrupt:
                    if modbus_client is not None and modbus_client.is_socket_open():
                        modbus_client.close()
                        logger.info("Closed PLC socket.")'''

            except Exception as e:
                # Server refuses connection or no internet. The first case might occur often, so FOR now, we do not
                # include it to the file logger.
                logger.debug(f"{e}")
                logger.debug(f"Sleeping for {PLC_connect_interval} seconds..")
                time.sleep(PLC_connect_interval)
                break

    except KeyboardInterrupt:  # Ctrl ^C event
        logger.info("Client ends process.")

    # We should keep whether the setpoint was given by the predictions or if the predictions did not reach the PLC.
    print('Writing to file.')

    csv_file = f'C:/Users/res4b/Desktop/modbus_tcp_ip/ml_control/setpoints_{dataframe_date}.csv'

    # If file does not exist aka the day has changed, and we need a new .csv, create it
    # Write row to DataFrame.csv
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writeheader()

    # Append the new row of information
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writerow(row)