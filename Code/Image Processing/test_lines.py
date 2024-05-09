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

def calculate_distance_from_origin(line):
    x1, y1, x2, y2 = line[0]
    distance_1 = np.sqrt(x1**2 + y1**2)
    distance_2 = np.sqrt(x2**2 + y2**2)
    return min(distance_1, distance_2)

def process(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de mediana para reducir el ruido
    median_blur_image = cv2.medianBlur(gray_image, 9)
    
    canny_image = cv2.Canny(median_blur_image, 150, 200)

    lines = cv2.HoughLinesP(canny_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=230,
                            lines=np.array([]),
                            minLineLength=290,
                            maxLineGap=100)
    
    filtered_lines = []
    
    if lines is not None:
        for line in lines:
            distance = calculate_distance_from_origin(line)
            # Filtrar las líneas que están a una distancia específica del origen
            if distance > 200 and distance < 1000:  # Ajusta los valores según sea necesario
                filtered_lines.append(line)
        
        image_with_lines = draw_the_lines(image, filtered_lines)
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
