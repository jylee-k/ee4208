#we will use these libraries
import cv2
import numpy as np
from scipy import ndimage

def roberts_edge_detection(image, threshold=10):
  """
  Applies Roberts edge detection to a grayscale image with optional thresholding.

  Args:
      image (np.ndarray): Grayscale image as a NumPy array.
      threshold (int, optional): Threshold value for edge detection. Defaults to 100.

  Returns:
      tuple: A tuple containing the original image, Roberts edge-detected image, and
              thresholded edge-detected image (if threshold is provided).
  """

  # Roberts kernels for vertical and horizontal edges
  roberts_cross_v = np.array([[1, 0], [0, -1]])
  roberts_cross_h = np.array([[0, 1], [-1, 0]])

  # Convert image to float64 for better precision
  image = image.astype('float64') / 255.0  # Normalize image (0-1 range)

  # Apply convolution for vertical and horizontal edges
  vertical = ndimage.convolve(image, roberts_cross_v)
  horizontal = ndimage.convolve(image, roberts_cross_h)

  # Calculate magnitude using hypotenuse (more accurate than squared sum)
  edged_img = np.sqrt(vertical**2 + horizontal**2)

  # Normalize and convert back to uint8 for display
  edged_img *= 255.0  # Scale back to 0-255 range
  edged_img = edged_img.astype('uint8')

  # Apply thresholding (optional)
  if threshold > 0:
    thresholded_edges = np.where(edged_img > threshold, 255, 0)
  else:
    thresholded_edges = edged_img  # No thresholding applied

  return image, horizontal, thresholded_edges

# Load the image
image = cv2.imread("motion_compensation_output/deblurred_denoised_smoothed_395.jpg", 0)  # Read as grayscale

# Apply Roberts edge detection with optional thresholding
original_img, edges_img, thresholded_edges = roberts_edge_detection(image, threshold=20)  # Adjust threshold as needed

# Display all images (original, edges, thresholded) in one window
cv2.imshow("Original vs. Roberts Edges vs. Threshold", np.hstack((original_img, edges_img, thresholded_edges)))
cv2.waitKey(0)
cv2.destroyAllWindows()