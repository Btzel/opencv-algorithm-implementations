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
out = cv2.VideoWriter(os.path.join(script_dir, '../external/datasets/random/walking_output.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30,(w,h))
ret,frame = cap.read()

#create a float numpy array with frame values
average = np.float32(frame)

while True:
    #get frame
    ret,frame = cap.read()
    if ret:
        #0.01 is the weight of image, play around to see how it changes
        cv2.accumulateWeighted(frame,average,0.01)
        #scales, calculates absolute values, and converts the result to 8-bit
        background = cv2.convertScaleAbs(average)
        display(frame,"frame")
        display(background,"background")
        out.write(background)
    else:
        break

cap.release()
out.release()