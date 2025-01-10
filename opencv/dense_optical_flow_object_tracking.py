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
#load video stream
cap = cv2.VideoCapture(os.path.join(script_dir, '../external/datasets/random/walking.avi'))

#width, height of the frame
width = int(cap.get(3))
height = int(cap.get(4))

#create the codec and the videowriter object
out = cv2.VideoWriter(os.path.join(script_dir, '../external/datasets/random/dense_optical_flow_walking.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30,(width,height))

#get first frame
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[...,1] = 255




while True:
    ret, frame = cap.read()

    if ret:
        next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #computes the dense optical flow using gunnar farnesbacks algorithm
        flow = cv2.calcOpticalFlowFarneback(prev_gray,next,
                                            None,0.5,3,15,3,5,1.2,0)
        
        #use flow to calculate the magnitude (speed) and angle of motion
        #use these values to calculate the color to reflect speed and angle
        magnitude,angle = cv2.cartToPolar(flow[...,0],flow[...,1])
        hsv[...,0] = angle * (100/(np.pi/2))
        hsv[...,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
        final = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        #save video
        out.write(final)

        #store current image as previous image
        prev_gray = next

    else:
        break
cap.release()
out.release()
