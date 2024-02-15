import socket
import time
import os

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

# Start handling connection

server_host = os.environ.get('Server_ip_navgreen_control')
server_port = int(os.environ.get('Server_port_navgreen_control'))

try:
    while True:

        # time.sleep(5*60)
        time.sleep(10)

        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            client_socket.connect((server_host, server_port))
            logger.info(f"Connected to {server_host}:{server_port}.")

            setpoint_DHW = float(client_socket.recv(1024).decode('utf-8'))
            logger.info(f"Setpoint from server: {setpoint_DHW}.")

            client_socket.close()

        except Exception as e:
            logger.error(f"Error connecting to the server: {e}")

except KeyboardInterrupt:
    logger.info("Client ends process.")