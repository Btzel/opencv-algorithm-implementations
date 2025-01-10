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
cap = cv2.VideoCapture(os.path.join(script_dir, '../external/datasets/random/slow.flv'))
#take first frame of the video
ret,frame = cap.read()
#get the h and w of the frame (required to be an integer)
width = int(cap.get(3))
height = int(cap.get(4))
#define the codec and create videowriter object.
#the output is stored in '*.avi' file
out = cv2.VideoWriter(os.path.join(script_dir, '../external/datasets/random/car_tracking_mean_shift.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30,(width,height))
#setup initial location of window
r,h,c,w = 250,90,400,125 #simply hardcoded values
track_window = (c,r,w,h)
#set up the roi for tracking
roi = frame[r:r+h,c:c+w]
hsv_roi = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT,10,1)
while True:
    ret,frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        #apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst,track_window,term_crit)
        #draw it on image
        x,y,w,h = track_window
        image2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(100,255,255),2)
        out.write(image2)
    else:
        break

cap.release()
out.release()