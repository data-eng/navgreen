import socket
import time
import os
import struct

from pymodbus.client import ModbusTcpClient

import logging

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s')
# Set log file, its level and format
file_handler = logging.FileHandler('./remote_control_logger.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
# Set stream its level and format
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
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


if __name__ == "__main__":

    # IP and port of server for communication
    server_host = os.environ.get('Server_ip_navgreen_control')
    server_port = int(os.environ.get('Server_port_navgreen_control'))

    # PLC IP address, port and connection intervals
    plc_ip = os.environ.get('Plc_ip')
    plc_port = 502
    reconnect_interval = 30  # Seconds
    PLC_connect_interval = 5 * 60

    too_cold = False
    min_setpoint, max_setpoint = 15.0, 49.0

    try:
        while True:
            # Try to connect with the server to get the new DHW - setpoint value
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                client_socket.connect((server_host, server_port))
                logger.debug(f"Connected to {server_host}:{server_port}.")

                setpoint_range = f"{min_setpoint}-{max_setpoint}"
                client_socket.sendall(setpoint_range.encode('utf-8'))
                logger.debug(f"Sent the setpoint range to the server : [{min_setpoint}, {max_setpoint}].")

                setpoint_DHW = client_socket.recv(1024).decode('utf-8')
                logger.info(f"Setpoint from server: {setpoint_DHW}.")

                client_socket.close()

                # Sometimes server may send unexpected values i.e. Ctrl ^C etc.
                try:
                    setpoint_DHW = float(setpoint_DHW)
                except Exception as e:
                    logger.error(f"{e}")
                    logger.debug(f"Sleeping for {reconnect_interval} seconds..")
                    time.sleep(reconnect_interval)
                    continue

                # Now, write the new setpoint to the PLC
                # Connect with the PLC
                modbus_client = None

                try:
                    while True:
                        try:
                            # Create a Modbus TCP/IP modbus_client
                            modbus_client = ModbusTcpClient(plc_ip, port=plc_port)
                            modbus_client.connect()

                            logger.debug("Acquire and write data.")

                            # Read holding registers
                            result = modbus_client.read_holding_registers(500, 20)
                            other_registers = modbus_client.read_holding_registers(542,
                                                                                   25)  # Read pressure and temp registers
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
                                        if DHW_buffer > 50.0:
                                            logger.debug('DHW_buffer too hot.. closing')
                                            # Turn of HP
                                            new_DHW_setpoint = min_setpoint

                                        write_value = True
                                    else:
                                        # If the hp is going to be turned on check that it respects the constraints
                                        if new_DHW_setpoint > DHW_buffer + 1:  # Start hp (+1 dt)
                                            if DHW_buffer <= 50.0 and BTES_TANK > 8.0:
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
                        logger.info("Closed PLC socket.")

                # PLC must write no sooner than 5 minutes
                time.sleep(5 * 60)

            except Exception as e:
                # Server refuses connection or no internet. The first case might occur often, so FOR noe, we do not
                # include it to the file logger
                logger.debug(f"{e}")
                logger.debug(f"Sleeping for {PLC_connect_interval} seconds..")
                time.sleep(PLC_connect_interval)

    except KeyboardInterrupt: # Ctrl ^C event
        logger.info("Client ends process.")
