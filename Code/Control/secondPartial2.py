import cv2  # Importa la biblioteca OpenCV para procesamiento de imágenes
import json  # Importa la biblioteca JSON para manejar datos en formato JSON
from queue import Queue  # Importa la clase Queue de la biblioteca estándar para manejar una cola de datos
import RPi.GPIO as GPIO  # Importa la biblioteca RPi.GPIO para controlar los pines GPIO en una Raspberry Pi
from pyzbar.pyzbar import decode  # Importa la función decode de la biblioteca pyzbar para decodificar códigos QR
import threading  # Importa la biblioteca threading para ejecutar múltiples tareas en paralelo
import numpy as np  # Importa la biblioteca NumPy para operaciones numéricas avanzadas
import collections

GPIO.setwarnings(False)  # Desactiva las advertencias de GPIO

# Se crea una cola global o thread-safe para almacenar los datos generados
data_queue = Queue()

with open('lineDetectionROI.json', 'r') as file:
    lineROICoords = json.load(file)

# Carga las coordenadas de la región de interés (ROI) para detección de códigos QR desde un archivo JSON
with open('qrDetectionROI.json', 'r') as file:
    qrROICoords = json.load(file)

# Function to process ROI 1 and generate data
def processOnROI1(roi4Lines,roi_center):
    # Realiza operaciones de procesamiento de imagen en ROI 1 y genera una salida
    output1 = roi4Lines.mean()  # Ejemplo de cálculo de salida
    data_queue.put(('roi1', output1))  # Almacena la salida con identificador 'roi1'
    processed_image = detectLines(roi4Lines,roi_center)  # Procesa el ROI 1 para detectar líneas
    
    # Verifica si la imagen procesada es válida
    if processed_image is not None:
        cv2.imshow('LinesDetection', processed_image)
    else:
        print("Error: No se pudo procesar la imagen.")
        
# Función para procesar ROI 2 y generar datos
def processOnROI2(roi4QR):
    # Realiza operaciones de procesamiento de imagen en ROI 2 y genera una salida
    output2 = roi4QR.max()  # Ejemplo de cálculo de salida
    data_queue.put(('roi2', output2))  # Almacena la salida con identificador 'roi2'

    
IN1 = 17  # Pin GPIO para controlar un motor (adelante izquierda)
IN2 = 22  # Pin GPIO para controlar un motor (adelante izquierda)
IN3 = 6   # Pin GPIO para controlar un motor (atras izquierda)
IN4 = 5   # Pin GPIO para controlar un motor (atras izquierda)
IN5 = 24  # Pin GPIO para controlar un motor (adelante derecha)
IN6 = 23  # Pin GPIO para controlar un motor (adelante derecha)
IN7 = 20  # Pin GPIO para controlar un motor (atras derecha)
IN8 = 21  # Pin GPIO para controlar un motor (atras derecha)



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

# Define motor pins
left_motor_pins = [IN1, IN2, IN3, IN4]  # Pins for left motor (forward and backward)
right_motor_pins = [IN5, IN6, IN7, IN8]  # Pins for right motor (forward and backward)

# Initialize PWM for left and right motors
left_motor_pwm = GPIO.PWM(left_motor_pins[0], 100)  # Assuming IN1 controls the speed of the left motor
right_motor_pwm = GPIO.PWM(right_motor_pins[0], 100)  # Assuming IN5 controls the speed of the right motor

# Start PWM
left_motor_pwm.start(0)
right_motor_pwm.start(0)


# Función para decodificar un código QR en la imagen capturada
def decodeQR():
    data = decode(frame)  # Decodifica el código QR en la imagen actual
    if data:  # Si se detecta un código QR
        value = data[0].data.decode("utf-8")  # Obtiene el valor del código QR
        print("Encoded Data Result:", value)
        # Si el valor es '1', activa los motores
        if value == '1':
            # Inicia y espera a que finalicen las subrutinas para activar los motores
            MoveForward.start()
            MoveForward.join()
            
    else:  # Si no se detecta ningún código QR, detiene los motores
        Stop.start()
        Stop.join()
        print("Waiting for qr code")
        

def detectLines(roiFrame, roi_center_x):
    
    # Converting image to gray scale
    grayScale = cv2.cvtColor(roiFrame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("GrayScale Image",grayScale)
    
    # median_blur_image = cv2.medianBlur(grayScale, 9)
    
    # Binarize the grayscale image (thresholding)
    _, binaryImage = cv2.threshold(grayScale, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary Image",binaryImage)
    
    # Invert the binary image
    invertedImage = cv2.bitwise_not(binaryImage)
    cv2.imshow("Inverted Image", invertedImage)
    
    # Smoothed image
    blurImage = cv2.GaussianBlur(invertedImage,(5,5),0)
    # Apply edge detector
    edgesImage = cv2.Canny(blurImage, 50, 150)
    # Hough transform line detection
    lines = cv2.HoughLines(edgesImage, 1, np.pi/180, 40)
    
    positive_rhos = collections.deque(maxlen=30)
    positive_thetas = collections.deque(maxlen=30)
    negative_rhos = collections.deque(maxlen=30)
    negative_thetas = collections.deque(maxlen=30)
    # Display all detected lines
    if lines is not None:
        for line in lines:
            rho, theta = line[0]  # Accessing rho and theta values of the line
            if rho >= 0:  # Positive line
                positive_rhos.append(rho)
                positive_thetas.append(theta)
            else:  # Negative line
                negative_rhos.append(rho)
                negative_thetas.append(theta)
            print(f"Line Parameters - rho: {rho}, theta: {theta}")

   
        
    if len(positive_rhos) > 0 and len(positive_thetas) > 0:
        avg_positive_rho = int(np.mean(positive_rhos))
        avg_positive_theta = np.mean(positive_thetas)
        a = np.cos(avg_positive_theta)
        b = np.sin(avg_positive_theta)
        x0_positive = a * avg_positive_rho
        y0_positive = b * avg_positive_rho
        x1_positive = int(x0_positive + 1000 * (-b))
        y1_positive = int(y0_positive + 1000 * (a))
        x2_positive = int(x0_positive - 1000 * (-b))
        y2_positive = int(y0_positive - 1000 * (a))
        cv2.line(roiFrame, (x1_positive, y1_positive), (x2_positive, y2_positive), (0, 255, 0), 20)
        # Calculate slope and y-intercept of positive line
        m_positive = (y2_positive - y1_positive) / (x2_positive - x1_positive)
        c_positive = y1_positive - m_positive * x1_positive

    # Calculate average rho and theta for negative lines
    if len(negative_rhos) > 0 and len(negative_thetas) > 0:
        avg_negative_rho = int(np.mean(negative_rhos))
        avg_negative_theta = np.mean(negative_thetas)
        a = np.cos(avg_negative_theta)
        b = np.sin(avg_negative_theta)
        x0_negative = a * avg_negative_rho
        y0_negative = b * avg_negative_rho
        x1_negative = int(x0_negative + 1000 * (-b))
        y1_negative = int(y0_negative + 1000 * (a))
        x2_negative = int(x0_negative - 1000 * (-b))
        y2_negative = int(y0_negative - 1000 * (a))
        cv2.line(roiFrame, (x1_negative, y1_negative), (x2_negative, y2_negative), (0, 0, 255), 20)
        # Calculate slope and y-intercept of negative line
        m_negative = (y2_negative - y1_negative) / (x2_negative - x1_negative)
        c_negative = y1_negative - m_negative * x1_negative

    if len(positive_rhos) > 0 and len(positive_thetas) > 0 and len(negative_rhos) > 0 and len(negative_thetas) > 0:
        # Calculate intersection point in x
        intersection_x = (c_negative - c_positive) / (m_positive - m_negative)

        # Draw vertical line at intersection point
        cv2.line(roiFrame, (int(intersection_x), 0), (int(intersection_x), roiFrame.shape[0]), (255, 0, 0), 5)
        cv2.line(roiFrame, (int(roi_center_x), 0), (int(roi_center_x), roiFrame.shape[0]), (255, 0, 255), 5)
        difference_x = roi_center_x - intersection_x
        controlAlgorithm(intersection_x,roi_center_x)
    return roiFrame


    
def controlAlgorithm(xInt,xRoi):
    constSpeed = 1
    if xRoi<xInt:
        correction=(np.abs(xRoi-xInt)*constSpeed)/xRoi
        
        leftSpeed=0
        # rightSpeed=constSpeed-correction
        rightSpeed = constSpeed
        print('Derecha')
    elif xRoi> xInt:
        correction=((xRoi-xInt)*constSpeed)/xRoi
        # leftSpeed=constSpeed-correction
        leftSpeed = constSpeed
        rightSpeed=0
        print('Izquirda')
    else:
        leftSpeed=constSpeed
        rightSpeed=constSpeed
    print(correction)    
    forward(leftSpeed, rightSpeed)
    

# Función para avanzar (encender los motores en dirección hacia adelante)
def forward(leftSpeed, rightSpeed):    
    left_pwm_speed = int(leftSpeed * 100)  # Convert speed to PWM duty cycle (0-100% range)
    right_pwm_speed = int(rightSpeed * 100)  # Convert speed to PWM duty cycle (0-100% range)
    
    # Set the appropriate GPIO pins to move the robot forward with the specified speeds
    GPIO.output(IN1, GPIO.LOW)  # Adelante izquierda
    GPIO.output(IN2, GPIO.HIGH)   # Adelante izquierda
    GPIO.output(IN3, GPIO.LOW)  # Atrás izquierda
    GPIO.output(IN4, GPIO.HIGH)   # Atrás izquierda
    GPIO.output(IN5, GPIO.LOW)  # Adelante derecha
    GPIO.output(IN6, GPIO.HIGH)   # Adelante derecha
    GPIO.output(IN7, GPIO.LOW)  # Atrás derecha
    GPIO.output(IN8, GPIO.HIGH)   # Atrás derecha
    
    # Set PWM duty cycle for left and right motors
    left_motor_pwm.ChangeDutyCycle(left_pwm_speed)
    right_motor_pwm.ChangeDutyCycle(right_pwm_speed)
    
    
   

# Función para detener (apagar los motores)
def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    GPIO.output(IN5, GPIO.LOW)
    GPIO.output(IN6, GPIO.LOW)
    GPIO.output(IN7, GPIO.LOW)
    GPIO.output(IN8, GPIO.LOW)
 

def open_camera(max_attempts=10):
    for i in range(max_attempts):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Camera opened successfully
            return cap

        print(f"Attempt {i + 1}: Camera not detected on device {i}")

    # No camera found within the attempts
    print("Error: Could not open camera on any available devices.")
    return None

# Attempt to open the camera using the function
cap = open_camera()

if cap is None:
    # Handle the case where no camera is found
    print("Exiting program due to camera detection failure.")
    exit()
    

# Bucle principal para la captura y procesamiento de los fotogramas
while True:
    # Lee un fotograma de la cámara
    ret, frame = cap.read()
    
    # Define las regiones de interés (ROI) para detección de líneas y códigos QR
    xLn, yLn, widthLn, heightLn = lineROICoords['x'], lineROICoords['y'], lineROICoords['width'], lineROICoords['height']
    linesROI = frame[yLn:yLn+heightLn, xLn:xLn+widthLn]
    
    xQR, yQR, widthQR, heightQR = qrROICoords['x'], qrROICoords['y'], qrROICoords['width'], qrROICoords['height']
    qrROI = frame[yQR:yQR+heightQR, xQR:xQR+widthQR]
    roi_center_x = xLn + widthLn // 2
    
    # Verifica si el fotograma se leyó correctamente
    if not ret:
        print("Error: Could not read frame.")
        break

    # Crea hilos para procesar cada ROI (Región de Interés)
    thread1 = threading.Thread(target=processOnROI1, args=(linesROI,roi_center_x,))
    # thread2 = threading.Thread(target=processOnROI2, args=(qrROI,))
   
    # Inicia ambos hilos
    thread1.start()
    # thread2.start()
   
    # Espera a que ambos hilos finalicen
    thread1.join()
    # thread2.join()
    
    # Muestra el fotograma de la región de interés para la detección de códigos QR
    # cv2.imshow("QR Scanning View", qrROI)
    
    # Define hilos para detener o avanzar los motores en base a la detección de QR
    
    # MoveForward=threading.Thread(target=forward,args=(leftSpeed, rightSpeed))
    Stop=threading.Thread(target=stop)
    
    # Intenta iniciar un hilo para decodificar el código QR en el fotograma actual
    # try:
    #     decoding = threading.Thread(target=decodeQR())
    #     decoding.start()
    # except Exception as e:
    #     print("An error occurred during QR code detection:", str(e))
    
    # Recupera las salidas de la cola y las almacena en un diccionario
    outputs = {}
    while not data_queue.empty():
        roi, output = data_queue.get()
        outputs[roi] = output
    
    # Verifica si se presiona la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
