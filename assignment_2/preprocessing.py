import os
import cv2
import numpy as np
from bm3d import bm3d

def calculate_average_noise_std(folder_path, roi_coords):
    """
    Calculate the average noise standard deviation from a folder of images.

    Parameters:
        folder_path (str): Path to the folder containing image files.
        roi_coords (tuple): Coordinates of the region of interest (ROI) (x1, y1, x2, y2).

    Returns:
        float: Average noise standard deviation.
    """
    # Initialize a list to store individual noise standard deviations
    noise_std_devs = []

    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    # Iterate through each image file
    for image_file in image_files:
        # Load the image
        image = cv2.imread(os.path.join(folder_path, image_file), cv2.IMREAD_GRAYSCALE)

        # Extract the ROI from the image
        x1, y1, x2, y2 = roi_coords
        roi = image[y1:y2, x1:x2]

        # Compute the standard deviation of pixel values in the ROI
        noise_std_dev = np.std(roi)

        # Append the computed standard deviation to the list
        noise_std_devs.append(noise_std_dev)

    # Calculate the average noise standard deviation
    average_noise_std_dev = np.mean(noise_std_devs)
    return average_noise_std_dev

def temporal_smoothing(input_folder, output_folder, window_size):
    """
    Apply temporal smoothing to a sequence of images stored in a folder.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where smoothed images will be saved.
        window_size (int): Number of frames to use for temporal averaging.

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

    # Iterate through each image file
    for i in range(len(image_files)):
        # Read the current frame
        current_frame = cv2.imread(os.path.join(input_folder, image_files[i]))
        smoothed_frame = np.zeros_like(current_frame, dtype=np.float32)

        # Initialize variables to keep track of valid frames in the window
        valid_frames = 0

        # Iterate through the window of frames centered around the current frame
        for j in range(max(0, i - window_size // 2), min(len(image_files), i + window_size // 2 + 1)):
            frame = cv2.imread(os.path.join(input_folder, image_files[j]))
            smoothed_frame += frame.astype(np.float32)
            valid_frames += 1

        # Perform temporal averaging
        smoothed_frame /= valid_frames
        smoothed_frame = smoothed_frame.astype(np.uint8)

        #denoised_frame = bm3d(smoothed_frame, sigma_psd=13.4)

        # Save the smoothed frame to the output folder
        cv2.imwrite(os.path.join(output_folder, f"smoothed_{image_files[i]}"), smoothed_frame)
        #cv2.imwrite(os.path.join(output_folder2, f"denoised_{image_files[i]}"), denoised_frame)

    print("Temporal smoothing completed.")

def bm3d_denoise(input_folder, output_folder, sigma_psd):
    """
    Apply temporal smoothing to a sequence of images stored in a folder.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where smoothed images will be saved.
        window_size (int): Number of frames to use for temporal averaging.

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

    # Iterate through each image file
    for i in range(len(image_files)):
        # Read the current frame
        current_frame = cv2.imread(os.path.join(input_folder, image_files[i]))

        denoised_frame = bm3d(current_frame, sigma_psd=sigma_psd)

        # Save the denoised frame to the output folder
        cv2.imwrite(os.path.join(output_folder, f"denoised_{image_files[i]}"), denoised_frame)

    print("Bm3d denoising completed.")

# Example usage
input_folder = "retina2"
smooth_folder = "retina2_smoothed"
bm3d_on_original = "retina2_denoise"
bm3d_plus_smooth_folder = "retina2_smoothed_plus_denoise"
window_size = 7  # Adjust this parameter to change the size of the temporal window
roi_coords = (250, 250, 340, 340)  # Example ROI coordinates (x1, y1, x2, y2)

temporal_smoothing(input_folder, smooth_folder, window_size)

#bm3d on original
orig_average_noise_std_dev = calculate_average_noise_std(input_folder, roi_coords)
bm3d_denoise(input_folder, bm3d_on_original, orig_average_noise_std_dev)

#bm3d on smoothing 
smooth_average_noise_std_dev = calculate_average_noise_std(smooth_folder, roi_coords)
bm3d_denoise(smooth_folder, bm3d_plus_smooth_folder, smooth_average_noise_std_dev)

bm3d_on_original_noise_sed_dev = calculate_average_noise_std(bm3d_on_original, roi_coords)
smooth_plus_bm3d_noise_std_dev = calculate_average_noise_std(bm3d_plus_smooth_folder, roi_coords)

print("Original Average Noise Standard Deviation:", orig_average_noise_std_dev)
print("Smoothing on original Average Noise Standard Deviation:", smooth_average_noise_std_dev)
print("Bm3d on original Average Noise Standard Deviation:", bm3d_on_original_noise_sed_dev)
print("Bm3d plus smoothing Average Noise Standard Deviation:", smooth_plus_bm3d_noise_std_dev)



'''
(ee4208_proj1) paul@PAULdeMacBook-Air assignment_2 % python  preprocessing.py
Average Noise Standard Deviation: 6.213702415994948
(ee4208_proj1) paul@PAULdeMacBook-Air assignment_2 % python  preprocessing.py
Average Noise Standard Deviation: 13.399395205973835
'''