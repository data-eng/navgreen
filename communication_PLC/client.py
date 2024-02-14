import socket
import os

def connect_to_server(host, port, message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        print(f"try connecting to {host}:{port}..")
        client_socket.connect((host, port))
        print("connected!")
        print("sending..")
        client_socket.sendall(message.encode('utf-8'))
        print("sent.")
        answer = client_socket.recv(1024).decode('utf-8')
        print(f"answer from server: {answer}")
    except Exception as e:
        print(f"Error connecting to the server: {e}")
    finally:
        client_socket.close()


if __name__ == "__main__":

    server_host = os.environ.get('Server_ip_navgreen_control')
    server_port = int(os.environ.get('Server_port_navgreen_control'))

    connect_to_server(server_host, server_port, "Hello, server!")