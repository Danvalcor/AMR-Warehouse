import cv2
import numpy as np

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    # Obtener las dimensiones del frame
    height, width, _ = image.shape
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de mediana para reducir el ruido
    median_blur_image = cv2.medianBlur(gray_image, 9)
    
    canny_image = cv2.Canny(median_blur_image, 150, 200)
    cv2.imshow('canny',canny_image)
    lines = cv2.HoughLinesP(canny_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=230,
                            lines=np.array([]),
                            minLineLength=10,
                            maxLineGap=100)
    
    if lines is not None:
        image_with_lines = draw_the_lines(image, lines)
    else:
        image_with_lines = image

    return image_with_lines

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
