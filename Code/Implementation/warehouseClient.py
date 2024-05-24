import socket

msgFromClient="Dinamito"
serverAddress = "192.168.1.11"

bytesToSend=msgFromClient.encode('utf-8')

serverAddress=(serverAddress,2222)

bufferSize=1024

aspireClientSocket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

while True:
    dataFromServer = None
    aspireClientSocket.sendto(bytesToSend,serverAddress)

    dataFromServer,serverAddress=aspireClientSocket.recvfrom(bufferSize)

    dataFromServer = dataFromServer.decode('utf-8')
    if dataFromServer == "Connected":
        print('Conected')
        print("Ip Address from server:", serverAddress[0])
        print("Port to listen:",serverAddress[1])
        break
    else:
        print('Waiting on server')

print("Valid instructions are: Deliver or Quit")
print("Input 'q' to quit the program")
while True:
    userInput=input("Instruction: ")
    if userInput == 'q':
        break
    else:
        userInput=userInput.encode('utf-8')
        aspireClientSocket.sendto(userInput,serverAddress)
    
    dataFromserver,serverAddress=aspireClientSocket.recvfrom(bufferSize)
    dataFromserver=dataFromserver.decode('utf-8')
    print('Server Response:',dataFromserver)
    if dataFromServer == "Delivering":
        while True:
            print('Server Response:',dataFromserver)
            aspireClientSocket.sendto(userInput,serverAddress)
            dataFromserver,serverAddress=aspireClientSocket.recvfrom(bufferSize)
            dataFromserver=dataFromserver.decode('utf-8')
            if dataFromServer == "Home":
                break
            
    
