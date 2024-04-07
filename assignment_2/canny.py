import cv2
import numpy as np

image = cv2.imread("data/395.bmp", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 200, 250)

cv2.imshow("Original vs. Canny Edge Detection", np.hstack((image, edges)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2 as cv
# from matplotlib import pyplot as plt

# img = cv.imread('data/395.bmp', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# edges = cv.Canny(img,100,200)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()