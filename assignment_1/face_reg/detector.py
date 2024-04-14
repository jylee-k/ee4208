import cv2
import os
import numpy as np

# Get the directory path of the current script
path = os.path.dirname(os.path.abspath(__file__))

# Create and load the face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + r'\trainer\trainer.yml')

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(path + r"\classifier\face.xml")

# Initialize video capture object
video_capture = cv2.VideoCapture(0)

# Define font for text display
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Process each detected face
    for (x, y, w, h) in faces:
        # Predict the label for the face
        label, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        name_dict = {0: 'charles', 1: 'chenyang', 2: 'fayang', 3: 'jaron', 4: 'junyoung', 5: 'zexuan'}

        # Map the label number to a user-friendly name
        name = name_dict.get(label, "Unknown")  

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 0, 0), 2)

        # Add text for the predicted name and confidence
        cv2.putText(frame, f"{name} -- {confidence:.2f}", (x, y+h+20), font, 1.1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
