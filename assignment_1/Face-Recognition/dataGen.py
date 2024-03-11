import cv2
import os

# Get the directory path of the current script
path = os.path.dirname(os.path.abspath(__file__))

# Initialize video capture object for webcam
cam = cv2.VideoCapture(0)

# Load the face detector from the XML file
detector = cv2.CascadeClassifier(path + r'\Classifiers\face.xml')

# User ID input with informative prompt
user_id = input("Enter your ID for the dataset: ")

# Initialize frame counter and offset for image cropping
frame_count = 0
offset = 50

while True:
    # Read a frame from the webcam
    ret, frame = cam.read()

    # Detect faces in the color frame (no grayscale conversion)
    faces = detector.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(128, 128), flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Create a filename with user ID and frame count
        filename = f"dataset/face{user_id}_{frame_count}.jpg"

        # Crop the face region from the color frame (ensure it's within frame boundaries)
        face_roi = frame[max(y-offset, 0):min(y+h+offset, frame.shape[0]), max(x-offset, 0):min(x+w+offset, frame.shape[1])]

        # Resize the cropped face to 128x128
        resized_face = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_AREA)

        # Save the cropped color face image to the dataset folder
        cv2.imwrite(filename, resized_face)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x-offset, y-offset), (x+w+offset, y+h+offset), (225, 0, 0), 2)

        # Display the full color webcam frame with the rectangle
        cv2.imshow('im', frame)

        # Wait for a key press with a slight delay to avoid overwhelming processing
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # Exit the loop after capturing enough images
        if frame_count >= 20:
            # Release the webcam object and close windows
            cam.release()
            cv2.destroyAllWindows()

            # Informative message after capturing enough images
            print(f"Captured {frame_count} images for ID: {user_id}")
            break

        # Increment frame counter for image filename
        frame_count += 1
