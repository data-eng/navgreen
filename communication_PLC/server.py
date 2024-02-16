import socket
import os

# Server's host and listening port
server_host = '0.0.0.0'
server_port = os.environ.get('Server_port_navgreen_control')

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_host, server_port))

# Only one connection allowed
server_socket.listen(1)
print('Server listening on ' + str(server_host) + ':' + str(server_port))

try:
    setpoint_DHW = None
    confirmation = None

    while confirmation != "y":
        setpoint_DHW = input("Please enter the desired value: ")

        try:
            # Only accept numerical values
            setpoint_DHW = float(setpoint_DHW)

            while True:
                confirmation = input("Is the entered value the new DHW setpoint? (y/n): ").lower()

                if confirmation == "y":
                    print("Sending the value to the PLC when it connects.")
                    break
                elif confirmation == "n":
                    print("Re-enter DHW setpoint.")
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

         # Incoming connection
        client_socket, client_address = server_socket.accept()
        print('Connection established with' + str(client_address))

        try:
            # If is number and user is sure about their choices, send setpoint to client
            client_socket.send(str(setpoint_DHW).encode('utf-8'))
            print("Setpoint sent.")

        except Exception as e:
            print("\nSending the setpoint failed: " + str(e))

        finally:
            # Terminate connection
            client_socket.close()
            print('Closing connection with ' + str(client_address))

except KeyboardInterrupt:
    # User performed Ctrl ^C
    print("\nServer stops listening")
