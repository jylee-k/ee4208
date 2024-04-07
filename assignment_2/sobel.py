import cv2
import numpy as np

image = cv2.imread("motion_compensation_output/deblurred_denoised_smoothed_395.jpg", cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

cv2.imshow("Original", image)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.imshow("Sobel combined", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
