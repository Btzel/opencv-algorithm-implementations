import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/car.jpg'))

#define range of color hsv
lower1 = np.array([0,70,50])
upper1 = np.array([10,255,255])

lower2 = np.array([170,70,50])
upper2 = np.array([180,255,255])

lower3 = np.array([0,0,130])
upper3 = np.array([255,125,255])

#convert image from rgb/bgr to hsv so we easily filter
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#use inRange to capture only the values between lower & upper
mask1 = cv2.inRange(hsv_image,lower1,upper1)
mask2 = cv2.inRange(hsv_image,lower2,upper2)
mask3 = cv2.inRange(hsv_image,lower3,upper3)

mask = mask1 | mask2 | mask3
#perform bitwise AND on mask and our original frame
res = cv2.bitwise_and(image,image,mask=mask)

display(image)
display(mask)
display(res)