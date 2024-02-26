import cv2 
import numpy as np 
from scipy import ndimage 

roberts_cross_v = np.array( [[1, 0 ], 
							[0,-1 ]] ) 

roberts_cross_h = np.array( [[ 0, 1 ], 
							[ -1, 0 ]] ) 

img = cv2.imread("data/395.bmp", cv2.IMREAD_GRAYSCALE)

vertical = ndimage.convolve( img, roberts_cross_v ) 
horizontal = ndimage.convolve( img, roberts_cross_h ) 

edged_img = np.sqrt( np.square(horizontal) + np.square(vertical)) 
cv2.imshow("output", edged_img)
cv2.waitKey(0)
