import socket

msgFromClient="Dinamito"

bytesToSend=msgFromClient.encode('utf-8')

serverAddress=('192.168.50.1',2222)

bufferSize=1024

aspireClientSocket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

# aspireClientSocket.sendto(bytesToSend,serverAddress)

# dataFromServer,serverAddress=aspireClientSocket.recvfrom(bufferSize)

# dataFromServer=dataFromServer.decode('utf-8')

# print('Data send from serve',dataFromServer)
# print("Ip Address from server:", serverAddress[0])
# print("Port to listen:",serverAddress[1])

while True:
    userInput=input("Tell the server to inc or DEC its Counter variable:")
    userInput=userInput.encode('utf-8')
    aspireClientSocket.sendto(userInput,serverAddress)
    dataFromserver,serverAddress=aspireClientSocket.recvfrom(bufferSize)
    dataFromserver=dataFromserver.decode('utf-8')
    print('Server Counter Count:',dataFromserver)