import re
import os
import cv2
import csv
import math
import easyocr
import pandas as pd
import tensorflow as tf
import object_detection
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_local
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
from cv2 import resizeWindow
font=cv2.FONT_HERSHEY_SIMPLEX

x = input('Enter your jpg file: ')
setimg = "D:\\IA\\OCR\\OCR_result\\" + x
img = cv2.imread(setimg)
def orientation_correction(img, save_image = False):   
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    median_angle = np.median(angles)
    
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
grayImage2 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
finalImg = cv2.cvtColor(bw_scanner(grayImage2), cv2.COLOR_GRAY2RGB)
pho = "D:\\IA\\OCR\\OCR_result\\"+x+"-pre.jpg"
cv2.imwrite(pho,finalImg)

PATH_TO_CFG = "D:\\IA\\OCR\\Tensorflow\\workspace\\models\\model2\\pipeline.config"
paths = {'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',"model2")}
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-22')).expect_partial()

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

PATH_TO_LABELS = "D:\\IA\\OCR\\Tensorflow\\workspace\\annotations\\label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
IMAGE_PATH = pho

img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections


detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,
            min_score_thresh=.3,
            agnostic_mode=False)

odresult = "D:\\IA\\OCR\\OCR_result\\"+x+"-od.jpg"
cv2.imwrite(odresult,image_np_with_detections)
ocr_model = PaddleOCR(use_angle_cls=True, lang="chinese_cht",use_gpu=True)

detection_threshold = 0.3
image = image_np_with_detections
width = image.shape[1]
height = image.shape[0]
scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]

temp = []
for idx, box in enumerate(boxes):
    roi = box*[height, width, height, width]
    region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
    ocr_result = ocr_model.ocr(region)
    for result in ocr_result:
        print(result[1][0])
        temp.append(result[1][0])

#result2
chinese = re.compile(u'[\u4e00-\u9fa5]+')
eng = re.compile(u'[a-zA-Z]')

#print(temp)

# b = brand, m = total, p = payment method, t = time, id = id

def is_english_char(ch):
    if ord(ch) not in (97,122) and ord(ch) not in (65,90):
        return False
    return True

print("==================================================================================") 
print(temp)

totalChar=["HKD","$","."]
paymentMethod=["卡","八","octo","pus","MAS","VI","銀聯","Unio","現金","Alipay","用"]
dateCheck=["/","-", "："]
    
def loop (temp):
    brand_temp=True
    recipt_brand,recipt_total,recipt_paymethod,recipt_Id,recipt_date=[],[],[],[],[]
    
    for i in range(len(temp)):
        digitcheck="".join(x for x in temp[i] if x.isdigit())
        if any(c in temp[i] for c in totalChar):
            recipt_total.append(i)
            brand_temp=False
            continue
        if any(c in temp[i] for c in paymentMethod):
            recipt_paymethod.append(i)
            brand_temp=False
            continue
        if any(c in temp[i] for c in dateCheck):
            recipt_date.append(i)
            brand_temp=False
            continue
        if len(digitcheck)>=4 and len(temp[i])>4:
            recipt_Id.append(i)
            brand_temp=False
            
        if (brand_temp):
            recipt_brand.append(i)
            
    return recipt_brand,recipt_total,recipt_paymethod,recipt_Id,recipt_date

recipt_brand,recipt_total,recipt_paymethod,recipt_Id,recipt_date =loop(temp)

def fix(a):
    a2 =""
    if int(len(a))==1:
        for i in a:
            a = temp[int(i)]
    elif int(len(a)) >=1:
        for i in a:
            a2 += temp[int(i)]+" "
        return a2
    else:
        a = "Null"
    return a

new_date = re.sub(chinese,"", fix(recipt_date))
new_Id = re.sub(chinese,"", fix(recipt_Id))
new_total = re.sub(chinese,"", fix(recipt_total))

final_name = x
final_brand = fix(recipt_brand)
final_total = new_total
final_date = new_date
final_pay = fix(recipt_paymethod)
final_ID = new_Id

f = open("OCR_result.csv", "a", newline="", encoding="utf-8-sig")
tup = (final_name,final_brand,final_total,final_date,final_pay,final_ID)
w = csv.writer(f)
w.writerow(tup)
f.close()

print("==================================================================================") 
print("Photo: " + x) 
print("Brand: " + fix(recipt_brand) )
print("Total: " + new_total )
print("Time: " + new_date )
print("payment method: " + fix(recipt_paymethod))
print("Receipt ID: " + new_Id)