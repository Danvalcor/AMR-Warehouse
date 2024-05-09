#Codigo Lab
import cv2
import json
import numpy as np
import collections


# Load ROI coordinates from a JSON file
with open('lineDetectionROI.json', 'r') as file:
    lineDetectionROI = json.load(file)

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

# Initialize lists to store rho and theta values for positive and negative lines
positive_rhos = []
positive_thetas = []
negative_rhos = []
negative_thetas = []

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break
    # Extract ROI from the frame using the loaded coordinates
    x, y, width, height = lineDetectionROI['x'], lineDetectionROI['y'], lineDetectionROI['width'], lineDetectionROI['height']
    roiFrame = frame[y:y+height, x:x+width]
    roi_center_x = x + width // 2
    
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
        
    # Display results
    cv2.imshow("Detected Image", roiFrame)
    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
