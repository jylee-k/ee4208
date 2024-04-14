import cv2
import os
import numpy as np
from PIL import Image

name_dict = {'charles': 0, 'chenyang': 1, 'fayang': 2, 'jaron': 3, 'junyoung': 4, 'zexuan': 5}

# Get the directory path of the current script
path = os.path.dirname(os.path.abspath(__file__))

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the face detector cascade
cascadePath = path + r"\classifier\face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Path to the dataset containing the images
dataPath = path + r'\dataset\gray'

def get_images_and_labels(datapath):
    """
    Reads images and labels from the dataset.

    Args:
        datapath (str): Path to the dataset directory.

    Returns:
        tuple: (images, labels) where:
            images (list): List of face images as NumPy arrays.
            labels (list): List of corresponding labels for the images.
    """

    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
    images = []
    labels = []

    for image_path in image_paths:
        # Read image, convert to grayscale, and create NumPy array
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')

        # Extract label from filename
        label = label = os.path.splitext(os.path.basename(image_path))[0].split("_")[0]
        if label in name_dict:
            label = name_dict[label]
        else:
            print(f"Warning: {label} not found.")
            continue

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(image)

        # If faces are found, process them
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(label)
            # Optional: Display the detected faces during training
            # cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
            # print(label)
            # cv2.waitKey(10)

    return images, labels

# Get faces and labels from the dataset
images, labels = get_images_and_labels(dataPath)

# Show a test image (optional)
# cv2.imshow('test', images[0])
# cv2.waitKey(1)

# Train the recognizer
recognizer.train(images, np.array(labels))

# Save the trained model
recognizer.save(path + r'\trainer\trainer.yml')

# Close all open windows
cv2.destroyAllWindows()
