import math
import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_local
from PIL import Image

def orientation_correction(img, save_image = False): 
    # GrayScale Conversion for the Canny Algorithm  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    # Using Houghlines to detect lines
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    # Finding angle of lines in polar coordinates
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    # Getting the median angle
    median_angle = np.median(angles)
    
    # Rotating the image with this median angle
    img_rotated = ndimage.rotate(img, median_angle)
    
    return img_rotated


def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255


def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)
##

img = cv2.imread('D:\\IA\\OCR\\OCR_result\\18.jpg')
#img = orientation_correction(f_img)
grayImage2 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

bwGray = bw_scanner(grayImage2)
no_noise = noise_removal(bwGray)
no_noise1 = cv2.cvtColor(no_noise, cv2.COLOR_GRAY2RGB)
finalImg = cv2.cvtColor(bw_scanner(no_noise1), cv2.COLOR_GRAY2RGB)

cv2.imwrite("D:\\IA\\OCR\\OCR_result\\output1.jpg",finalImg)


