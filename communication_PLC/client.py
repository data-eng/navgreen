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


# def read_reg_value(register, index, multiplier):


if __name__ == "__main__":

    # IP and port of server for communication
    server_host = os.environ.get('Server_ip_navgreen_control')
    server_port = int(os.environ.get('Server_port_navgreen_control'))

    # PLC IP address, port and connection intervals
    plc_ip = os.environ.get('Plc_ip')
    plc_port = 502
    reconnect_interval = 30  # Seconds

    try:
        while True:
            # Try to connect with the server to get the new DHW - setpoint value
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                client_socket.connect((server_host, server_port))
                logger.info(f"Connected to {server_host}:{server_port}.")

                setpoint_DHW = float(client_socket.recv(1024).decode('utf-8'))
                logger.info(f"Setpoint from server: {setpoint_DHW}.")

                client_socket.close()

                if setpoint_DHW < 15 or setpoint_DHW > 51:
                    # Threshold for setpoint is [15, 51]
                    logger.info("Setpoint not in threshold [15, 51]. Not accepted.")
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

                            logger.info("Aquire and write data.")

                            # Read holding registers
                            result = modbus_client.read_holding_registers(500, 20)  # Read registers from 500 to 519
                            setpoint_registers = modbus_client.read_holding_registers(540,2)  # Read holding registers 540 and 541, setpoints
                            other_registers = modbus_client.read_holding_registers(542, 25)  # Read remaining pressure and temp registers
                            result_alarm = modbus_client.read_coils(8192 + 506, 1, int=0)

                            # Get the values we want
                            general_alarm = result_alarm.bits[0]
                            BTES_TANK = read_reg_value(result, 7, 10)
                            DHW_buffer = read_reg_value(result, 10, 10)
                            POWER_HP = read_reg_value(result, 19, 1000)
                            compressor_HZ = read_reg_value(other_registers, 24, 10)
                            T_setpoint_DHW_modbus = read_reg_value(setpoint_registers, 1, 10)

                            # Setpoint value given by the server
                            new_DHW_setpoint = setpoint_DHW

                            # Get ready to write:

                            # If no alarm is raised
                            if not general_alarm:

                                # If heat pump is on
                                if compressor_HZ >= 30 and POWER_HP > 2:
                                    if DHW_buffer > 52.0:
                                        # Turn of HP
                                        new_DHW_setpoint = 15

                                ############################################
                                # Write when conditions are met:

                                # - If heat pump is on and temperature of DHW tank (top layer) > 52 oC, hp must be turned off
                                # - If temp of the gorund (BTES tank) < 8 oC, hp must be turned off AND turned back on IF themp of ground tank is above 12 oC
                                # - not alarm!

                                # write(num_of_register, value)
                                # CONVERT IT TO INTEGER !!
                                # client.write_coil(8192+126, True)

                                ############################################

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
                time.sleep(10)
                # time.sleep(5*60)

            except Exception as e:
                logger.error(f"{e}")
                logger.info(f"Sleeping for {reconnect_interval} seconds..")
                time.sleep(reconnect_interval)

    except KeyboardInterrupt:
        logger.info("Client ends process.")
