import RPi.GPIO as GPIO
import time
import threading


GPIO.setwarnings(False)


# Definir los pines de control del driver L298N
IN1 = 17  # Pin Enable A
IN2 = 22  # Pin Input 2
IN3=6
IN4=5
IN5=24
IN6=23
IN7=20
IN8=21

# Configurar los pines de la Raspberry Pi
GPIO.setmode(GPIO.BCM)

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

GPIO.setup(IN5, GPIO.OUT)
GPIO.setup(IN6, GPIO.OUT)

GPIO.setup(IN7, GPIO.OUT)
GPIO.setup(IN8, GPIO.OUT)

# Función para mover el motor hacia adelante
def forward(input1,input2):
    
    GPIO.output(input2, GPIO.HIGH)
    GPIO.output(input1, GPIO.LOW)
    

# Función para detener el motor
def stop(input1,input2):
    GPIO.output(input1, GPIO.LOW)
    GPIO.output(input2, GPIO.LOW)

while True:
    # Mover el motor hacia adelante durante 2 segundos
    # M1=threading.Thread(target=forward,args=(IN1,IN2))
    # M2=threading.Thread(target=forward,args=(IN3,IN4))
    # M3=threading.Thread(target=forward,args=(IN5,IN6))
    M4=threading.Thread(target=forward,args=(IN7,IN8))
    M4Stop=threading.Thread(target=stop,args=(IN7,IN8))
    
    # M1.start()
    # M2.start()
    # M3.start()
    # M4.start()
    # M4.join()
    # time.sleep(2)
    
    M4Stop.start()
    # M1.join()
    # M2.join()
    # M3.join()
    M4Stop.join()
    
    


