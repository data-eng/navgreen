import socket
import os

def start_listening(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))

    server_socket.listen(1)
    print('Server listening on ' + str(host) + ':' + str(port))

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            print('Connection established with' + str(client_address))

            data = client_socket.recv(1024)
            print('Received message:' + str(data.decode('utf-8')))

            answer = "ok mate"
            client_socket.send(answer.encode('utf-8'))

            client_socket.close()

            print('closing connection with ' + str(client_address))

    except KeyboardInterrupt:
        print("Bye from the server")

if __name__ == "__main__":

    server_host = os.environ.get('Server_ip_navgreen_control')
    server_port = int(os.environ.get('Server_port_navgreen_control'))

    start_listening(server_host, server_port)

