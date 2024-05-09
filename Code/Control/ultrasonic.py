#NO BORRAR
#Codigo de prueba para funcionamiento de ultrasonico por separado

# import RPi.GPIO as GPIO
# import time

# GPIO.setmode(GPIO.BCM)
# trigPin = 4
# echoPin = 17

# GPIO.setup(trigPin, GPIO.OUT)
# GPIO.setup(echoPin, GPIO.IN)

# stopDistance=25

# try:
#     while True:
#         GPIO.output(trigPin, 0)
#         time.sleep(2E-6)
#         GPIO.output(trigPin, 1)
#         time.sleep(10E-6)
#         GPIO.output(trigPin, 0)

#         # Espera hasta que el pin echoPin se ponga en HIGH o hasta que pasen 0.1 segundos
#         start_time = time.time()
#         while GPIO.input(echoPin) == 0 and time.time() - start_time < 0.1:
#             pass

#         # Si el pin echoPin se pone en HIGH, mide el tiempo transcurrido
#         if GPIO.input(echoPin) == 1:
#             echoStartTime = time.time()
#             while GPIO.input(echoPin) == 1:
#                 pass
#             echoStopTime = time.time()

#             elapsedTime = echoStopTime - echoStartTime
#             currentMeasureDistance = (elapsedTime * 34300) / 2
#             if currentMeasureDistance < stopDistance:
#                 print("It worked")
#             else:
#                 print("distance:", currentMeasureDistance, "cm")


# except KeyboardInterrupt:
#     GPIO.cleanup()
#     print('GPIO Good to GO')



#Codigo con thread para el principal
import RPi.GPIO as GPIO
import time
import threading


GPIO.setwarnings(False)

trigPin = 4
echoPin = 17
stopDistance = 25  # Adjust as needed

GPIO.setmode(GPIO.BCM)
GPIO.setup(trigPin, GPIO.OUT)
GPIO.setup(echoPin, GPIO.IN)

current_distance = 0  # Variable global para almacenar la distancia actual
# distance_lock = threading.Lock()  # Lock para sincronizar el acceso a current_distance

def measure_distance(trigPin, echoPin, stopDistance):
    global current_distance
    GPIO.output(trigPin, 0)
    time.sleep(2E-6)
    GPIO.output(trigPin, 1)
    time.sleep(10E-6)
    GPIO.output(trigPin, 0)

    # Espera hasta que el pin echoPin se ponga en HIGH o hasta que pasen 0.1 segundos
    start_time = time.time()
    while GPIO.input(echoPin) == 0 and time.time() - start_time < 0.1:
        pass

    # Si el pin echoPin se pone en HIGH, mide el tiempo transcurrido
    if GPIO.input(echoPin) == 1:
        echoStartTime = time.time()
        while GPIO.input(echoPin) == 1:
            pass
        echoStopTime = time.time()

        elapsedTime = echoStopTime - echoStartTime
        current_distance = (elapsedTime * 34300) / 2


# Loop principal
while True:
    distance_thread = threading.Thread(target=measure_distance, args=(trigPin, echoPin, stopDistance))
    distance_thread.start()

    # with distance_lock:
    if current_distance < stopDistance:
        print("Distancia actual:", current_distance)
        print("Menor")
    else:
        print("Distancia actual:", current_distance)
        print("Mayor")
    time.sleep(1)  # Ajusta el tiempo de espera según sea necesari



