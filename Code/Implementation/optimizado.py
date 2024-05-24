import cv2  # Importa la biblioteca OpenCV para procesamiento de imágenes
import json  # Importa la biblioteca JSON para manejar datos en formato JSON
from queue import Queue  # Importa la clase Queue de la biblioteca estándar para manejar una cola de datos
import RPi.GPIO as GPIO  # Importa la biblioteca RPi.GPIO para controlar los pines GPIO en una Raspberry Pi
from pyzbar.pyzbar import decode  # Importa la función decode de la biblioteca pyzbar para decodificar códigos QR
import threading  # Importa la biblioteca threading para ejecutar múltiples tareas en paralelo
import numpy as np  # Importa la biblioteca NumPy para operaciones numéricas avanzadas
import collections
import time

# Function to process ROI 1 and generate data
def processOnROI1(roi4Lines,roi_center):
    # Realiza operaciones de procesamiento de imagen en ROI 1 y genera una salida
    processed_image = detectLines(roi4Lines,roi_center)  # Procesa el ROI 1 para detectar líneas 
    # Verifica si la imagen procesada es válida
    if processed_image is None:
        print("Error: No se pudo procesar la imagen.")
    else:
        data_queue.put(('roi1', processed_image))  # Almacena la salida con identificador 'roi1'

        
# Función para procesar ROI 2 y generar datos
def processOnROI2(roi4QR):
    # Realiza operaciones de procesamiento de imagen en ROI 2 y genera una salida
    data_queue.put(('roi2', roi4QR))  # Almacena la salida con identificador 'roi2'

# Función para decodificar un código QR en la imagen capturada
def decodeQR():
    data = decode(frame)  # Decodifica el código QR en la imagen actual
    if data:  # Si se detecta un código QR
        value = data[0].data.decode("utf-8")  # Obtiene el valor del código QR
        print("Encoded Data Result:", value)
        # Si el valor es '1', activa los motores
        if value == '1':
            print("In")

            # # Inicia y espera a que finalicen las subrutinas para activar los motores
            # MoveForward.start()
            # MoveForward.join()
            
    else:  # Si no se detecta ningún código QR, detiene los motores
        # Stop.start()
        # Stop.join()
        print("Waiting for qr code")
        

def calculate_line_parameters(lines, maxlen=30):
    positive_lines = collections.deque(maxlen=maxlen)
    negative_lines = collections.deque(maxlen=maxlen)

    for line in lines:
        rho, theta = line[0]
        m = np.tan(theta - np.pi/2)  # Restamos np.pi/2 para obtener la pendiente correcta
        # Filtra las líneas muy horizontales (pendiente cercana a cero)
        if abs(m) < 10* np.pi / 180:
            continue  # Descarta líneas muy horizontales
        elif rho >= 0:
            positive_lines.append((rho, theta))
        else:
            negative_lines.append((rho, theta))
    return {'positive': list(positive_lines), 'negative': list(negative_lines)}

def draw_lines(roiFrame, lines_info, thickness = 10):
    for key, lines in lines_info.items():
        if len(lines) > 0:
            color = (0, 0, 255) if key == 'negative' else (255, 0, 0)
            length = 10000 if key == 'negative' else 1000
            for rho, theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = int(a * rho)
                y0 = int(b * rho)
                x1 = int(x0 + length * (-b)) 
                y1 = int(y0 + length * (a)) 
                x2 = int(x0 - length * (-b)) 
                y2 = int(y0 - length * (a)) 
                cv2.line(roiFrame, (x1, y1), (x2, y2), color, thickness)

    return roiFrame

def detectLines(roiFrame, roi_center_x):
    _, binaryImage = cv2.threshold(cv2.cvtColor(roiFrame, cv2.COLOR_BGR2GRAY), 80, 255, cv2.THRESH_BINARY_INV)
    blurImage = cv2.GaussianBlur(binaryImage, (5, 5), 0) 
    edgesImage = cv2.Canny(blurImage, 50, 150)
    lines = cv2.HoughLines(edgesImage, 1, np.pi/180, 40)
    
    if lines is not None:
        lines_info = calculate_line_parameters(lines)
        #roiFrame = draw_lines(roiFrame, lines_info, thickness=2)
        m = {}
        c = {}
        for key, lines in lines_info.items():
            if lines:
                rhos, thetas = zip(*lines)
                avg_rho = np.mean(rhos)
                avg_theta = np.mean(thetas)
                x0 = avg_rho * np.cos(avg_theta)
                y0 = avg_rho * np.sin(avg_theta)
                length = 10000 if key == 'negative' else 1000
                x1 = int(x0 + length * (-np.sin(avg_theta)))
                y1 = int(y0 + length * (np.cos(avg_theta)))
                x2 = int(x0 - length * (-np.sin(avg_theta)))
                y2 = int(y0 - length * (np.cos(avg_theta)))
                cv2.line(roiFrame, (x1, y1), (x2, y2), (0, 255, 0), 10)
                m[key] = (y2 - y1) / (x2 - x1)
                c[key] = y1 - m[key] * x1
        
        if len(m) > 1:
            intersection_x = (c['negative'] - c['positive']) / (m['positive'] - m['negative'])
            cv2.line(roiFrame, (int(intersection_x), 0), (int(roi_center_x), roiFrame.shape[0]), (255, 0, 255), 5)
            cv2.circle(roiFrame, (int(intersection_x), 0), 20, (255, 0, 255), -1)
            print(intersection_x)
            controlAlgorithm(intersection_x, roi_center_x)

    return roiFrame



def controlAlgorithm(xInt,xRoi, constSpeed = 100):
    if xRoi<xInt:
        correction=(np.abs(xRoi-xInt)*constSpeed)/xRoi
        leftSpeed=constSpeed
        rightSpeed = constSpeed-correction
        print(f"Center: {xRoi}, Int: {xInt}, Derecha: {correction}")

    elif xRoi> xInt:
        correction=((xRoi-xInt)*constSpeed)/xRoi
        rightSpeed=constSpeed
        leftSpeed = constSpeed-correction
        print(f"Center: {xRoi}, Int: {xInt}, Izquierda: {correction}")
    else:
        leftSpeed=constSpeed
        rightSpeed=constSpeed
    forward(leftSpeed, rightSpeed)

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
    
# Función para detener (apagar los motores)
def stop():
     #front left
    GPIO.output(IN1, GPIO.LOW)  
    GPIO.output(IN2, GPIO.LOW)
    # front right
    GPIO.output(IN3, GPIO.LOW)  
    GPIO.output(IN4, GPIO.LOW)
    # back left
    GPIO.output(IN5, GPIO.LOW)  
    GPIO.output(IN6, GPIO.LOW)   
    # Back right
    GPIO.output(IN7, GPIO.LOW) 
    GPIO.output(IN8, GPIO.LOW)

# Función para avanzar (encender los motores en dirección hacia adelante)
def forward(leftSpeed, rightSpeed):    
    left_pwm_speed = int(leftSpeed)  # Convert speed to PWM duty cycle (0-100% range)
    right_pwm_speed = int(rightSpeed)  # Convert speed to PWM duty cycle (0-100% range)
    
    # Set the approO.priate GPIO pins to move the robot forward with the specified speeds
    # Front left
    GPIO.output(IN1, GPIO.HIGH)  
    GPIO.output(IN2, GPIO.LOW)
    # front right
    GPIO.output(IN3, GPIO.HIGH)  
    GPIO.output(IN4, GPIO.LOW)
    # back left
    GPIO.output(IN5, GPIO.HIGH)  
    GPIO.output(IN6, GPIO.LOW)   
    # Back right
    GPIO.output(IN7, GPIO.HIGH) 
    GPIO.output(IN8, GPIO.LOW)   
    print("left speed: ",left_pwm_speed," -----  right speed: ",right_pwm_speed)
    # Set PWM duty cycle for forward
    ENFI_pwm.ChangeDutyCycle(100)
    ENFD_pwm.ChangeDutyCycle(100)
    ENBI_pwm.ChangeDutyCycle(left_pwm_speed)
    ENBD_pwm.ChangeDutyCycle(right_pwm_speed*.3) # Compensamos la velocidad 

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
    
if __name__ == "__main__":
    
    # Variable declarations.
    IN1 = 22  # Pin GPIO para controlar un motor (frente izquierda hacia atras )
    IN2 = 25  # Pin GPIO para controlar un motor (frente izquierda hacia adelante  ) 
    IN3 = 5   # Pin GPIO para controlar un motor (adelante derecha)
    IN4 = 6   # Pin GPIO para controlar un motor (adelante derecha)
    ENFI = 2
    ENFD = 3
    IN5 = 20  # Pin GPIO para controlar un motor (atras izquierda)
    IN6 = 21  # Pin GPIO para controlar un motor (atras izquierda)
    IN7 = 24  # Pin GPIO para controlar un motor (atras derecha)
    IN8 = 23  # Pin GPIO para controlar un motor (atras derecha)
    ENBI = 19
    ENBD=26
    GPIO.setwarnings(False)

    trigPin = 4 #Ultrasonico trigger
    echoPin = 17 #ultrasonico echo
    stopDistance = 25  # Adjust as needed
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(trigPin, GPIO.OUT)
    GPIO.setup(echoPin, GPIO.IN)

    # GPIO pinouts configurations
    GPIO.setwarnings(False)  # Desactiva las advertencias de GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(IN5, GPIO.OUT)
    GPIO.setup(IN6, GPIO.OUT)
    GPIO.setup(IN7, GPIO.OUT)
    GPIO.setup(IN8, GPIO.OUT)
    GPIO.setup(ENFI, GPIO.OUT)
    GPIO.setup(ENFD, GPIO.OUT)
    GPIO.setup(ENBI, GPIO.OUT)
    GPIO.setup(ENBD, GPIO.OUT)
    # Define motor pins
    left_motor_pins = [IN1, IN5]  # Pins for left motor (forward and backward)
    right_motor_pins = [IN3, IN7]  # Pins for right motor (forward and backward)

    GPIO.output(IN2, GPIO.LOW) 
    GPIO.output(IN6, GPIO.LOW) 
    GPIO.output(IN4, GPIO.LOW) 
    GPIO.output(IN8, GPIO.LOW) 

    ENFI_pwm=GPIO.PWM(ENFI,1000)
    ENFI_pwm.start(0)
    ENFD_pwm=GPIO.PWM(ENFD,1000)
    ENFD_pwm.start(0)
    ENBI_pwm=GPIO.PWM(ENBI,1000)
    ENBI_pwm.start(0)
    ENBD_pwm=GPIO.PWM(ENBD,1000)
    ENBD_pwm.start(0)
    # Initialize PWM for left and right motors
    #left_motor_pwm = [GPIO.PWM(pin, 100) for pin in left_motor_pins]  # Pins 0 and 2 for left motor (forward and backward)
    #right_motor_pwm = [GPIO.PWM(pin, 100) for pin in right_motor_pins]  # Pins 0 and 2 for right motor (forward and backward)

    # Se crea una cola global o thread-safe para almacenar los datos generados
    data_queue = Queue()
    
    # Load ROIs
    with open('lineDetectionROI.json', 'r') as file:
        lineROICoords = json.load(file)

    # Carga las coordenadas de la región de interés (ROI) para detección de códigos QR desde un archivo JSON
    with open('qrDetectionROI.json', 'r') as file:
        qrROICoords = json.load(file)
        

    # Attempt to open the camera using the function
    cap = open_camera(5)

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
        linesROI = frame[yLn:yLn+heightLn, xLn:xLn+widthLn].copy()
        #linesROI = frame #Pruebas con imagenes
	    
        xQR, yQR, widthQR, heightQR = qrROICoords['x'], qrROICoords['y'], qrROICoords['width'], qrROICoords['height']
        qrROI = frame[yQR:yQR+heightQR, xQR:xQR+widthQR].copy()
        roi_center_x = linesROI.shape[1]/2  #Pruebas con imagenes
	    
	    # Verifica si el fotograma se leyó correctamente
        if not ret:
            print("Error: Could not read frame.")
            break
               
         # Define hilos para detener o avanzar los motores en base a la detección de QR
        Stop=threading.Thread(target=stop)

	    # Crea hilos para procesar cada ROI (Región de Interés)
        thread1 = threading.Thread(target=processOnROI1, args=(linesROI,roi_center_x,))
        thread2 = threading.Thread(target=processOnROI2, args=(qrROI,))
    
	    # Inicia ambos hilos
        thread1.start()
        thread2.start()
	    # Espera a que ambos hilos finalicen
        thread1.join()
        thread2.join()

       # MoveForward=threading.Thread(target=forward,args=(1, 1))
	    
        distance_thread = threading.Thread(target=measure_distance, args=(trigPin, echoPin, stopDistance))
        distance_thread.start()
     
	    # Intenta iniciar un hilo para decodificar el código QR en el fotograma actual
        try:
            decoding = threading.Thread(target=decodeQR())
            decoding.start()
        except Exception as e:
            print("An error occurred during QR code detection:", str(e))

        # ultrasonicData=measure_distance(trigPin, echoPin, stopDistance)
        if current_distance < stopDistance:
            print("It worked")
            Stop.start()
            Stop.join()
        else:
            thread_running = True 
            print("distance:", current_distance, "cm")
           # MoveForward.start()
           # MoveForward.join()
  	
  	    # Recupera las salidas de la cola y las almacena en un diccionario
        outputs = {}
        while not data_queue.empty():
            roi, output = data_queue.get()
            outputs[roi] = output
        scale = 0.2
        # Muestra todos los outputs almacenados en el diccionario
        for roi, output in outputs.items():
            height, width = frame.shape[:2]
            resized_output = cv2.resize(output, (int(width * scale), int(height *scale)))
            #cv2.imshow(roi, resized_output)
            
        # Muestra la imagen original a tamano reducido
        height, width = frame.shape[:2]
        resized_output = cv2.resize(output, (int(width * scale), int(height *scale)))
        #cv2.imshow("Original view", resized_output)
        
	# Verifica si se presiona la tecla 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

	# Libera la cámara y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()
    Stop.start()
    Stop.join()

