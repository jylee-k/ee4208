import cv2
import numpy as np
import os

def motion_compensation(image, motion):
    """
    Compensates for motion blur in an image using estimated motion.
    
    Args:
    - image: Input image (numpy array).
    - motion: Estimated motion vector (x, y).
    
    Returns:
    - Motion-compensated image.
    """
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, -motion[0]], [0, 1, -motion[1]]])
    motion_compensated = cv2.warpAffine(image, M, (cols, rows))
    return motion_compensated

def estimate_motion(image1, image2):
    """
    Estimates motion between two images using phase correlation.
    
    Args:
    - image1: First input image (numpy array).
    - image2: Second input image (numpy array).
    
    Returns:
    - Estimated motion vector (x, y).
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Compute phase correlation
    motion_fft = cv2.phaseCorrelate(gray1, gray2)
    motion = motion_fft[0]
    
    return motion


def deblur_image(image, kernel_size=(15, 15)):
    """
    Deblurs an image using Gaussian blur.
    
    Args:
    - image: Input image (numpy array).
    - kernel_size: Size of the Gaussian kernel for blurring.
    
    Returns:
    - Deblurred image.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    deblurred = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return deblurred

def deblur_images_in_folder(input_folder, output_folder):
    """
    Deblurs images in a folder by compensating for motion blur and saves them in the output folder.
    
    Args:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to the folder to store deblurred images.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of image files in folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    for i in range(len(image_files)-1):
        # Load consecutive images
        image1 = cv2.imread(os.path.join(input_folder, image_files[i]))
        image2 = cv2.imread(os.path.join(input_folder, image_files[i+1]))
        
        # Estimate motion between consecutive images
        motion = estimate_motion(image1, image2)
        
        # Compensate for motion blur
        motion_compensated = motion_compensation(image2, motion)
        
        # Deblur the compensated image
        deblurred = deblur_image(motion_compensated)
        
        # Save deblurred image in output folder
        cv2.imwrite(os.path.join(output_folder, f'deblurred_{i}.jpg'), deblurred)

# Example usage
input_folder = 'retina2_smoothed_plus_denoise'
output_folder = 'motion_compensation_output'
deblur_images_in_folder(input_folder, output_folder)
