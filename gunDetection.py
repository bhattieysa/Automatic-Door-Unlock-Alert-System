import numpy as np
import cv2
import imutils

# Load the cascade classifier for knife detection
gun_cascade = cv2.CascadeClassifier('knifecascade.xml')

# Open the camera
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = camera.read()
    
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect knives in the frame
    knives = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(150, 150))
    
    # Process each detected knife
    for (x, y, w, h) in knives:
        aspect_ratio = float(w) / h
        # Adjust these thresholds based on the expected aspect ratio of a knife
        if 0.5 < aspect_ratio < 2.0 and w > 50 and h > 50:
            # Draw a rectangle around the detected knife
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Extract the region of interest (ROI) for further analysis if needed
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

    # Display the frame with knife detections
    cv2.imshow('Knife Detection', frame)
    
    # Check for user input to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()