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
    linesROI = process(roi4Lines)  # Procesa el ROI 1 para detectar líneas
    cv2.imshow('LinesDetection',linesROI)

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
def forward(input1, input2):
    GPIO.output(input2, GPIO.HIGH)
    GPIO.output(input1, GPIO.LOW)

# Función para detener (apagar los motores)
def stop(input1, input2):
    GPIO.output(input2, GPIO.LOW)
    GPIO.output(input1, GPIO.LOW)

# Función para decodificar un código QR en la imagen capturada
def decodeQR():
    data = decode(frame)  # Decodifica el código QR en la imagen actual
    if data:  # Si se detecta un código QR
        value = data[0].data.decode("utf-8")  # Obtiene el valor del código QR
        print("Encoded Data Result:", value)
        # Si el valor es '1', activa los motores
        if value == '1':
            # Inicia y espera a que finalicen las subrutinas para activar los motores
            M1Forward.start()
            M2Forward.start()
            M3Forward.start()
            M4Forward.start()
            M1Forward.join()
            M2Forward.join()
            M3Forward.join()
            M4Forward.join()
    else:  # Si no se detecta ningún código QR, detiene los motores
        M1Stop.start()
        M2Stop.start()
        M3Stop.start()
        M4Stop.start()
        M1Stop.join()
        M2Stop.join()
        M3Stop.join()
        M4Stop.join()
        print("Waiting for qr code")
        
def draw_the_lines(img, lines):
    # Copia la imagen de entrada para evitar modificar la original
    img = np.copy(img)
    # Crea una imagen en blanco del mismo tamaño que la imagen de entrada
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Dibuja cada línea detectada en la imagen en blanco
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    # Combina la imagen original con la imagen en blanco para resaltar las líneas
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    # Obtener las dimensiones del frame
    height, width, _ = image.shape
    
    # Convierte la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica un filtro de mediana para reducir el ruido
    median_blur_image = cv2.medianBlur(gray_image, 9)
    
    # Detecta bordes en la imagen usando el algoritmo de Canny
    canny_image = cv2.Canny(median_blur_image, 150, 200)
    cv2.imshow('canny',canny_image)
    
    # Detecta líneas en la imagen utilizando la transformada de Hough
    lines = cv2.HoughLinesP(canny_image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=80,
                            lines=np.array([]),
                            minLineLength=3,
                            maxLineGap=20)
    
    # Si se detectan líneas
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                m = (y2 - y1) / (x2 - x1)  # Calcula la pendiente de la línea
                b = y1  # Calcula la ordenada al origen de la línea
                print("y = {}x + {}".format(m, b))  # Imprime la ecuación de la línea
        # Dibuja las líneas detectadas en la imagen
        image_with_lines = draw_the_lines(image, lines)
    else:
        image_with_lines = image  # Si no se detectan líneas, se conserva la imagen original
    
    return image_with_lines




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
    M1Stop=threading.Thread(target=stop,args=(IN1,IN2))
    M2Stop=threading.Thread(target=stop,args=(IN3,IN4))
    M3Stop=threading.Thread(target=stop,args=(IN5,IN6))
    M4Stop=threading.Thread(target=stop,args=(IN7,IN8))
    
    M1Forward=threading.Thread(target=forward,args=(IN1,IN2))
    M2Forward=threading.Thread(target=forward,args=(IN3,IN4))
    M3Forward=threading.Thread(target=forward,args=(IN5,IN6))
    M4Forward=threading.Thread(target=forward,args=(IN7,IN8))
    
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
