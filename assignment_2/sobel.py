import cv2
import numpy as np

image = cv2.imread("data/395.bmp", cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
combined_sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

cv2.imshow("Original", image)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.imshow("Sobel combined", combined_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img = cv.imread('data/395.bmp', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# ksize = 3
# gX = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize)
# gY = cv.Sobel(img, ddepth=cv.CV_32F, dx=0, dy=1, ksize=ksize)
# # the gradient magnitude images are now of the floating point data
# # type, so we need to take care to convert them back a to unsigned
# # 8-bit integer representation so other OpenCV functions can operate
# # on them and visualize them
# gX = cv.convertScaleAbs(gX)
# gY = cv.convertScaleAbs(gY)
# # combine the gradient representations into a single image
# combined = cv.addWeighted(gX, 0.5, gY, 0.5, 0)
# # show our output images
# cv.imshow("Sobel/Scharr X", gX)
# cv.imshow("Sobel/Scharr Y", gY)
# cv.imshow("Sobel/Scharr Combined", combined)
# cv.waitKey(0)
