#we will use these libraries
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

#a function to calculate edges of given image path
def roberts_cross(img_path, threshold=15.5):
    # Read the image using OpenCV
    #Reads the image from the specified path in grayscale, simplifying processing.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Define the Roberts Cross kernels for horizontal and vertical edge detection.
    gx = np.array([[1, 0], [0, -1]])
    gy = np.array([[0, 1], [-1, 0]])
    
    # Apply convolution operations to calculate horizontal and vertical gradients.
    gradient_x = ndimage.convolve(image, gx)
    gradient_y = ndimage.convolve(image, gy)
    
    # Compute the magnitude of the gradient using the square root of the sum of squares.
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Apply a threshold to create a binary mask, where pixels with magnitudes exceeding the threshold are considered part of an edge.
    edges = magnitude > threshold
    
    return edges

# Example usage
image_path = 'data/395.bmp'
edge_images = []
for i in range(2, 6):
    edge_images.append(roberts_cross(image_path, threshold=13+i/3))

# Display the original and edge-detected images
original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Plotting the images using matplotlib
plt.figure(figsize=(20, 10))

# Original Image
plt.subplot(1, 5, 1)
#Display the original image using matplotlib
plt.imshow(original_image_rgb)
plt.title('Original Image')
plt.axis('off')

for i in range(2, 6):
    
    # Edge-detected Image
    plt.subplot(1, 5, i)
    #Display the edge-detected image using matplotlib with a grayscale color map.
    plt.imshow(edge_images[i-2], cmap='gray')
    plt.title('Edge-detected Image')
    plt.axis('off')

# Show the plot containing both images.
plt.show()