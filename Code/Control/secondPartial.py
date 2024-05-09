import cv2  # Importa la biblioteca OpenCV para procesamiento de imágenes
import json  # Importa la biblioteca JSON para manejar datos en formato JSON
from queue import Queue  # Importa la clase Queue de la biblioteca estándar para manejar una cola de datos
import RPi.GPIO as GPIO  # Importa la biblioteca RPi.GPIO para controlar los pines GPIO en una Raspberry Pi
from pyzbar.pyzbar import decode  # Importa la función decode de la biblioteca pyzbar para decodificar códigos QR
import threading  # Importa la biblioteca threading para ejecutar múltiples tareas en paralelo
import numpy as np  # Importa la biblioteca NumPy para operaciones numéricas avanzadas

GPIO.setwarnings(False)  # Desactiva las advertencias de GPIO

# Se crea una cola global o thread-safe para almacenar los datos generados
data_queue = Queue()

with open('lineDetectionROI.json', 'r') as file:
    lineROICoords = json.load(file)

# Carga las coordenadas de la región de interés (ROI) para detección de códigos QR desde un archivo JSON
with open('qrDetectionROI.json', 'r') as file:
    qrROICoords = json.load(file)

# Function to process ROI 1 and generate data
def processOnROI1(roi4Lines):
    # Realiza operaciones de procesamiento de imagen en ROI 1 y genera una salida
    output1 = roi4Lines.mean()  # Ejemplo de cálculo de salida
    data_queue.put(('roi1', output1))  # Almacena la salida con identificador 'roi1'
    processed_image = detectLines(roi4Lines)  # Procesa el ROI 1 para detectar líneas
    
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



# Función para avanzar (encender los motores en dirección hacia adelante)
def forward(leftSpeed, rightSpeed):    
    # Set the PWM duty cycle for each motor based on the calculated speeds
    left_pwm_speed = int(leftSpeed * 255)  # Convert speed to PWM duty cycle (0-255 range)
    right_pwm_speed = int(rightSpeed * 255)  # Convert speed to PWM duty cycle (0-255 range)
    
    # Set the appropriate GPIO pins to move the robot forward with the specified speeds
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    GPIO.output(IN5, GPIO.HIGH)
    GPIO.output(IN6, GPIO.LOW)
    GPIO.output(IN7, GPIO.HIGH)
    GPIO.output(IN8, GPIO.LOW)
    
    
   

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
        

def detectLines(image):
    # Obtener las dimensiones del frame
    xLn, yLn, widthLn, heightLn = lineROICoords['x'], lineROICoords['y'], lineROICoords['width'], lineROICoords['height']
    
    # Calculate center point of ROI
    roi_center_x = xLn + widthLn // 2
    
    # Convierte la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica un filtro de mediana para reducir el ruido
    median_blur_image = cv2.medianBlur(gray_image, 9)
    
    _, binaryImage = cv2.threshold(median_blur_image, 90, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary Image", binaryImage)

    # Invert the binary image
    invertedImage = cv2.bitwise_not(binaryImage)
    cv2.imshow("Inverted Image", invertedImage)
    
    # Smoothed image
    blurImage = cv2.GaussianBlur(invertedImage, (5, 5), 0)
    # Apply edge detector
    edgesImage = cv2.Canny(blurImage, 50, 150)
    # Hough transform line detection
    lines = cv2.HoughLines(edgesImage, 1, np.pi/180, 80)
    # Display all detected lines
    if lines is not None:
        for line in lines:
            rho, theta = line[0]  # Accessing rho and theta values of the line
            print(f"Line Parameters - rho: {rho}, theta: {theta}")

    myLineScale = 50
    numberOfLines = 1
    myLine = 0
    
    if lines is not None: 
        myRho, myTheta = lines[myLine][0]  # Accessing parameters of the line
        print(f"My Line - rho: {myRho}, theta: {myTheta}")

        # Draw the line on the image
        rho, theta = lines[myLine][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + myLineScale * (-b))
        y1 = int(y0 + myLineScale * (a))
        x2 = int(x0 - myLineScale * (-b))
        y2 = int(y0 - myLineScale * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        
        # Calculate intersection point with ROI
        intersection_x = int((rho - y0 * np.sin(theta)) / np.cos(theta))
        
        # Calculate difference between ROI center and intersection point
        difference_x = roi_center_x - intersection_x
        
        controlAlgorithm(intersection_x,roi_center_x)
    else:
        print(f"Less than {numberOfLines} lines detected.")
    # Display results
    return image


    
def controlAlgorithm(xInt,xRoi,constSpeed):
    if xRoi<xInt:
        correction=(np.abs(xRoi-xInt)*constSpeed)/xRoi
        leftSpeed=constSpeed-correction
        rightSpeed=constSpeed+correction
    elif xRoi> xInt:
        correction=((xRoi-xInt)*constSpeed)/xRoi
        leftSpeed=constSpeed+correction
        rightSpeed=constSpeed-correction
    else:
        leftSpeed=constSpeed
        rightSpeed=constSpeed
    
    return leftSpeed, rightSpeed
  
    



# Abre la cámara de vídeo (cámara índice 1) para captura
cap = cv2.VideoCapture(1)  # 0 para la cámara predeterminada

# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: Could not open camera.")
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
    
    # Verifica si el fotograma se leyó correctamente
    if not ret:
        print("Error: Could not read frame.")
        break

    # Crea hilos para procesar cada ROI (Región de Interés)
    thread1 = threading.Thread(target=processOnROI1, args=(linesROI,))
    thread2 = threading.Thread(target=processOnROI2, args=(qrROI,))
   
    # Inicia ambos hilos
    thread1.start()
    thread2.start()
   
    # Espera a que ambos hilos finalicen
    thread1.join()
    thread2.join()
    
    # Muestra el fotograma de la región de interés para la detección de códigos QR
    cv2.imshow("QR Scanning View", qrROI)
    
    # Define hilos para detener o avanzar los motores en base a la detección de QR
    
    MoveForward=threading.Thread(target=forward,args=(leftSpeed, rightSpeed))
    Stop=threading.Thread(target=stop)
    
    # Intenta iniciar un hilo para decodificar el código QR en el fotograma actual
    try:
        decoding = threading.Thread(target=decodeQR())
        decoding.start()
    except Exception as e:
        print("An error occurred during QR code detection:", str(e))
    
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
