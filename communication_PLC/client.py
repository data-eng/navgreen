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
    client.write_coil(register, new_value)

    return


if __name__ == "__main__":

    # IP and port of server for communication
    server_host = os.environ.get('Server_ip_navgreen_control')
    server_port = int(os.environ.get('Server_port_navgreen_control'))

    # PLC IP address, port and connection intervals
    plc_ip = os.environ.get('Plc_ip')
    plc_port = 502
    reconnect_interval = 30  # Seconds

    too_cold = False
    min_setpoint, max_setpoint = 15.0, 49.0

    try:
        while True:
            # Try to connect with the server to get the new DHW - setpoint value
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                client_socket.connect((server_host, server_port))
                logger.info(f"Connected to {server_host}:{server_port}.")

                setpoint_range = f"{min_setpoint}-{max_setpoint}"
                client_socket.sendall(setpoint_range.encode('utf-8'))
                logger.info(f"Sent the setpoint range to the server : [{min_setpoint}, {max_setpoint}].")

                setpoint_DHW = client_socket.recv(1024).decode('utf-8')
                logger.info(f"Setpoint from server: {setpoint_DHW}.")

                client_socket.close()

                # Sometimes server may send unexpected values i.e. Ctrl ^C etc.
                try:
                    setpoint_DHW = float(setpoint_DHW)
                except Exception as e:
                    logger.error(f"{e}")
                    logger.info(f"Sleeping for {reconnect_interval} seconds..")
                    time.sleep(reconnect_interval)
                    continue

                # Now, write the new setpoint to the PLC
                # Connect with the PLC
                modbus_client = None

                # This try-except is to catch a ctrl-c
                try:

                    while True:
                        print("still here")

                        try:
                            # Create a Modbus TCP/IP modbus_client
                            modbus_client = ModbusTcpClient(plc_ip, port=plc_port)
                            modbus_client.connect()

                            logger.info("Acquire and write data.")

                            # Read holding registers
                            result = modbus_client.read_holding_registers(500, 20)
                            setpoint_registers = modbus_client.read_holding_registers(540, 2)  # Read setpoints
                            other_registers = modbus_client.read_holding_registers(542, 25)  # Read pressure and temp registers
                            result_alarm = modbus_client.read_coils(8192 + 506, 1, int=0)

                            # Get the values we want
                            general_alarm = result_alarm.bits[0]
                            BTES_TANK = read_reg_value(result, 7, 10)
                            DHW_buffer = read_reg_value(result, 10, 10)
                            POWER_HP = read_reg_value(result, 19, 1000)
                            compressor_HZ = read_reg_value(other_registers, 24, 10)
                            # T_setpoint_DHW_modbus = read_reg_value(setpoint_registers, 1, 10)

                            # Setpoint value given by the server
                            new_DHW_setpoint = setpoint_DHW

                            # Get ready to write:

                            # If no alarm is raised
                            if not general_alarm:
                                # If the heatpump was not turned off previously due to cold temperature
                                if not too_cold:
                                    # If heat pump is on
                                    if compressor_HZ >= 30.0 and POWER_HP > 2.0:
                                        # Top of tank is too hot
                                        if DHW_buffer > 50.0:
                                            # Turn of HP
                                            new_DHW_setpoint = min_setpoint

                                        # Bottom of tank is too cold
                                        if BTES_TANK <= 8.0:
                                            too_cold = True
                                            new_DHW_setpoint = min_setpoint

                                    # Everything written regardless whether hp is on
                                    write_reg_value(modbus_client, 541, new_DHW_setpoint, 10)
                                else:
                                    # If hp was closed due to cold temperature, should wait until it is at least 12
                                    if BTES_TANK >= 12.0:
                                        too_cold = False
                                        new_DHW_setpoint = max_setpoint
                                        write_reg_value(modbus_client, 541, new_DHW_setpoint, 10)

                            break

                        # PLC exception raised
                        except Exception as e:
                            logger.error(f"{e}")
                            logger.info(f"Sleeping for {reconnect_interval} seconds..")
                            time.sleep(reconnect_interval)
                            continue

                # Close connection when interrupted by the user
                except KeyboardInterrupt:
                    if modbus_client is not None and modbus_client.is_socket_open():
                        modbus_client.close()
                        logger.info("Closed PLC socket.")

                # PLC must write no sooner than 5 minutes
                # time.sleep(10)
                time.sleep(5*60)

            except Exception as e:
                logger.error(f"{e}")
                logger.info(f"Sleeping for {reconnect_interval} seconds..")
                time.sleep(reconnect_interval)

    except KeyboardInterrupt:
        logger.info("Client ends process.")