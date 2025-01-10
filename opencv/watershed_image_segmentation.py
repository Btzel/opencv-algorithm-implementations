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
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/water_coins.jpg'))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#threshold using otsu
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#noise kernel
kernel = np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
#sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
#finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#marker labeling
#connected components determines the connectivity of block-like regions in a binary image.
ret,markers = cv2.connectedComponents(sure_fg)
#add one to all labels so that sure background is not 0, but 1
markers = markers+1
#now, mark the region of unknown with zero
markers[unknown==255] = 0
markers= cv2.watershed(image,markers)
image[markers == -1] = [255,0,0]

display(image)