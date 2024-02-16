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

                            # Read holding register D500, D501, D502
                            result_coils2 = modbus_client.read_coils(8192 + 509, 7, int=0)  # read coils from 509 to 515
                            result_coils = modbus_client.read_coils(8192 + 500, 4)  # Read coils from 500 to 503

                            result = modbus_client.read_holding_registers(500, 20)  # Read registers from 500 to 519
                            results_registers2 = modbus_client.read_holding_registers(520, 18)  # Read registers from 520 to 527
                            setpoint_registers = modbus_client.read_holding_registers(540, 2)  # Read holding registers 540 and 541, setpoint SH and setpoint DHW

                            other_registers = modbus_client.read_holding_registers(542, 25)  # Read remaining pressure and temp registers

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

                            ############################################
                            # Write when conditions are met:

                            # - If heat pump is on and temperature of DHW tank (top layer) > 52 oC, hp must be turned off
                            # - If temp of the gorund (BTES tank) < 8 oC, hp must be turned off AND turned back on IF themp of ground tank is above 12 oC

                            # write(num_of_register, value)
                            # CONVERT IT TO INTEGER !!
                            # client_modbus.write_coil(8192 + 126, True)

                            ############################################

                            # PLC must write no sooner than 5 minutes
                            time.sleep(10)
                            # time.sleep(5*60)

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

            except Exception as e:
                logger.error(f"{e}")
                logger.info(f"Sleeping for {reconnect_interval} seconds..")
                time.sleep(reconnect_interval)


    except KeyboardInterrupt:
        logger.info("Client ends process.")
