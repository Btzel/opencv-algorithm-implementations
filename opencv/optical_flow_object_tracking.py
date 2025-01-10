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
#load video stream
cap = cv2.VideoCapture(os.path.join(script_dir, '../external/datasets/random/walking.avi'))

#width, height of the frame
width = int(cap.get(3))
height = int(cap.get(4))

#create the codec and the videowriter object
out = cv2.VideoWriter(os.path.join(script_dir, '../external/datasets/random/optical_flow_walking.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30,(width,height))

#set parameters for ShiTomasi corner detection
feature_params = dict(
    maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7
)

#set parameters for Lucas Kanade optical flow
lucas_kanade_params = dict(
    winSize = (15,15),
    maxLevel = 4,
    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT,10,0.03)
)

#create some random colors
#used to create our trails and object movement int the image
color = np.random.randint(0,255,(100,3))

#take the first frame and find corners
ret,prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

#find initial corner locations
prev_corners = cv2.goodFeaturesToTrack(prev_gray,mask=None,**feature_params)

#create mask image for drawing purposes
mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()

    if ret:
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #calculate optical flow
        new_corners,status,errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                             frame_gray,
                                                             prev_corners,
                                                             None,
                                                             **lucas_kanade_params)
        
        #select and store good points
        good_new = new_corners[status==1]
        good_old = prev_corners[status==1]

        #draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(frame,mask)
        #save video
        out.write(img)
        
        #now update the previous frame and previous points
        prev_gray = frame_gray.copy()
        prev_corners = good_new.reshape(-1,1,2)

    else:
        break
cap.release()
out.release()