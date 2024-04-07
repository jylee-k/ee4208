import cv2
import numpy as np

image = cv2.imread("motion_compensation_output/deblurred_denoised_smoothed_395.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 100, 240)

cv2.imshow("Original vs. Canny Edge Detection", np.hstack((image, edges)))
cv2.waitKey(0)
cv2.destroyAllWindows()