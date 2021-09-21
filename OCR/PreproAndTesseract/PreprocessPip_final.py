import math
#import imutils
import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_local
#from matplotlib import pyplot as plt
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

# Importing img
import os.path 
import glob 
# img_dir = "" # Enter Directory of all images  
# data_path = os.path.join('C:\\IA\ocr-c\\Data','*.jpg') 

files = glob.glob('D:\\IA\\OCR\\OCR_result\\16.jpg')
#files = glob.glob('C:\\IA\ocr-c\\Data0.5\\*.jpg') 
data = [] 
for f1 in files: 
    img = cv2.imread(f1) 
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    basename = os.path.basename(f1) 
    name = os.path.splitext(basename)[0] 
    #finalImg = cv2.cvtColor(noise_removal(bw_scanner(gray)), cv2.COLOR_GRAY2RGB)
    #finalImg = bw_scanner(gray)
    finalImg = cv2.fastNlMeansDenoising(bw_scanner(gray),None,10,7,21)
    cv2.imwrite('D:\\IA\\OCR\\OCR_result\\' + name + '_final.jpg', finalImg)
    


#grayImage2 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

#finalImg = orientation_correction(cv2.cvtColor(noise_removal(bw_scanner(grayImage2)), cv2.COLOR_GRAY2RGB))

#grayImage1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#bwGray = bw_scanner(grayImage2)

#no_noise = noise_removal(bwGray)
#no_noise1 = cv2.cvtColor(no_noise, cv2.COLOR_GRAY2RGB)

#finalImg = orientation_correction(no_noise1)

#cv2.imwrite("Prepro/Data1/final.jpg",finalImg)