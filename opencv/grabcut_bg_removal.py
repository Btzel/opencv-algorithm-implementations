import sys
import cv2
import dlib
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
copy = image.copy()

mask = np.zeros(image.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

x1,y1,x2,y2 = 75,300,1200,750
start = (x1, y1)
end = (x2, y2)

rect = (x1,y1,x2-x1,y2-y1)

cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8')
image = image * mask2[:,:,np.newaxis]

display(mask * 80)
display(mask2 * 255)
display(image)