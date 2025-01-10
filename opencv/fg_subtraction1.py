import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(os.path.join(script_dir, '../external/datasets/random/walking_short_clip.mp4'))

#get height and width of the frame (required to be an integer)
w = int(cap.get(3))
h = int(cap.get(4))

#define the codec and create video writer object. the output is stored in '*.avi' file.
out = cv2.VideoWriter(os.path.join(script_dir, '../external/datasets/random/walking_output_GM.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30,(w,h))

#initialize background subtractor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
foreground_background = cv2.createBackgroundSubtractorKNN()

#loop once video is succesfuly loaded
while True:
    ret, frame = cap.read()

    if ret:
        #apply background subtractor to get our foreground mask
        foreground_mask = foreground_background.apply(frame)
        foreground_mask = cv2.morphologyEx(foreground_mask,cv2.MORPH_OPEN,kernel)
        
        display(foreground_mask)
    else:
        break

cap.release()
out.release()
