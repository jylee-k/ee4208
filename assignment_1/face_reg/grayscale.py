import cv2
import os

# Get the dataset directory path
dataset_path = "dataset"

# Create the "gray/" folder if it doesn't exist
gray_folder = os.path.join(dataset_path, "gray")
if not os.path.exists(gray_folder):
    os.makedirs(gray_folder)

# Loop through all files in the dataset directory
for filename in os.listdir(dataset_path):
    # Check if the file is a JPG image
    if filename.endswith(".jpg"):
        # Construct the full path to the image file
        image_path = os.path.join(dataset_path, filename)

        # Read the color image
        image = cv2.imread(image_path)

        # Convert the color image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Construct the new filename for the grayscale image
        grayscale_filename = os.path.join(gray_folder, filename)  # Save in "gray/" folder

        # Save the grayscale image
        cv2.imwrite(grayscale_filename, grayscale_image)

        # Print informative message
        print(f"Converted {filename} to grayscale and saved to gray folder")
