import cv2
import numpy as np

def detect_lines_in_roi(roi_coordinates):
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        
        
        # Converting image to gray scale
        grayScale = cv2.cvtColor(roiFrame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("GrayScale Image", grayScale)
        
        median_blur_image = cv2.medianBlur(grayScale, 9)
        
        # Binarize the grayscale image (thresholding)
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
            cv2.line(roiFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            print(f"Less than {numberOfLines} lines detected.")
        # Display results
        cv2.imshow("Detected Image", roiFrame)
        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# Define ROI coordinates
roi_coordinates = (x, y, width, height)  # Define these values
# Call the function with ROI coordinates
detect_lines_in_roi(roi_coordinates)
