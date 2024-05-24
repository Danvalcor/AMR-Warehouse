import socket
import logging
import time

## Log definition
# Log name and format definition.
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_filename = f"session_{timestamp}.log"
log_format = '%(asctime)s - %(levelname)s - %(client_ip)s - %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)


## UDP comunication

# Define a buffer for data communication.
bufferSize = 1024

# Set the commununication port.
serverPort = 2222

# Set server IP address.
serverAddress = "192.168.50.1"
serverAddress = "192.168.1.11"

# Set up communication protocol
raspBerryPiSocket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

# Link the socket with its info
raspBerryPiSocket.bind((serverAddress,serverPort))
print("Server is up and listening...")

clientMessage = None
running = True
while running: 
    # Sit and wait for a client to send a message.
    clientMessage, clientAddress = raspBerryPiSocket.recvfrom(bufferSize)

    # Decode the message
    clientMessage = clientMessage.decode('utf-8')
    
    # Display the message
    print(clientMessage)
    print(f"Client Address: {clientAddress[0]}")
    
    # Act upon the message. 

    if clientMessage == "Deliver":
        # Call main
        # Server Response
        message = "Delivering"
    if clientAddress == "Quit":
        # My response back to the client.
        message = "Powering Off..."
        running = False
    else:
        message = "Not a valid option."
        
    # Response to server
    bytesToSend = message.encode('utf-8')
    raspBerryPiSocket.sendto(bytesToSend, clientAddress)
    
print("Shutting down server...")
