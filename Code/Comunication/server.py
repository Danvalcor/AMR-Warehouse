import socket
import time

# Estrategia de comunicacion UDP.

# Define a buffer for data communication.
bufferSize = 1024

# Define Messageto send to the client.
msgFromServer = "My Frase"

# Set the commununication port.
serverPort = 2222

# Set server IP address.
serverAddress = "198.168.4.3"

# Fromat the data you will sent to the client.
bytesToSend = msgFromServer.encode('utf-8')

# Set up communication protocol
raspBerryPiSocket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

# Link the socket with its info
raspBerryPiSocket.bind((serverAddress,serverPort))
print("Server si up and listening...")

# Sit and wait for a client to send a message.
clientMessage, clientAddress = raspBerryPiSocket.recvfrom(bufferSize)

# Decode the message
clientAddress = clientAddress.decode('utf-8')
# Display the message
print(clientMessage)
print(f"Client Address: {clientAddress[0]}")

# My response back to the client.
raspBerryPiSocket.sendto(bytesToSend, clientAddress[0])