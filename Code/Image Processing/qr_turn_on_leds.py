
import cv2
import RPi.GPIO as GPIO 
from pyzbar.pyzbar import decode
import threading

GPIO.setwarnings(False)
cap = cv2.VideoCapture(0) # The zero is for the default camera
qrDetector = cv2.QRCodeDetector() #QR-Code detector object


GPIO.setmode(GPIO.BCM)

# Definir los pines de control del driver L298N
IN1 = 17  # Pin Enable A
IN2 = 22  # Pin Input 2



GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

def forward(input1,input2):
    GPIO.output(input2, GPIO.HIGH)
    GPIO.output(input1, GPIO.LOW)
    
def stop(input1,input2):
    GPIO.output(input2, GPIO.LOW)
    GPIO.output(input1, GPIO.LOW)
# Validate if camera sensor opened as expected
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        print("Error: Could not read a frame")
        break
    
    M1Stop=threading.Thread(target=stop,args=(IN1,IN2),daemon=True)
    M1Forward=threading.Thread(target=forward,args=(IN1,IN2),daemon=True)
    
    try:
        data= decode(frame)
        if data:
            value=data[0].data.decode("utf-8")
            print("Encoded Data Result: ",value)
            if value == '1':
                print("entro")
                M1Forward.start()
                M1Forward.join()
            
        else:
            M1Stop.start()
            M1Stop.join()
            print("Waiting for qr code")
            
    except Exception as e:
        print("An error ocurred during QR code detection: ", str(e))
    
    # Display the rame in a window
    cv2.imshow("Camera View", frame)
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q'):
        M1Stop.join()
        M1Forward.join()

        break
cap.release()
cv2.destroyAllWindows()
