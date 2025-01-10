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
#define array range of color hsv for yellow
lower = np.array([22, 200, 200])
upper = np.array([30, 255, 255])

#create an empty array to store points
points = []

#load video stream and get the default sizes h w
cap = cv2.VideoCapture(os.path.join(script_dir, '../external/datasets/random/object_tracking1.mp4'))
width = int(cap.get(3))
height = int(cap.get(4))
                            
#define codec and create videowriter object
out = cv2.VideoWriter(os.path.join(script_dir, '../external/datasets/random/color_object_tracking.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30,(width,height))

ret,frame = cap.read()
Height,Width = frame.shape[:2]
frame_count = 0
radius = 0

while True:
    ret,frame = cap.read()
    if ret:
        hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #threshold the hsv image to get only yellow color
        mask = cv2.inRange(hsv_img,lower,upper)
    
        contours,_ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame,contours,-1,(0,255,0),thickness=1)
        #create empty centre array to store the centroid of mass
        center = int(Height/2),int(Width/2)

        if len(contours) > 0:
            #get the largest counter and its center
            c = max(contours,key=cv2.contourArea)
            (x,y),radius = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            #sometimes small contours of a point will couse a division by zero error
            try:
                if M['m00'] != 0:
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                else:
                    center = (int(Height / 2), int(Width / 2))
                    print("Zero division avoided: using default center")
            except KeyError as e:
                center = (int(Height / 2), int(Width / 2))
                print(f"Key error: {e}. Default center used.")
            except Exception as e:
                center = (int(Height / 2), int(Width / 2))
                print(f"Unexpected error: {e}. Default center used.")

            #draw circle
            
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,0,255),2)
            cv2.circle(frame, center, 5, (0,255,0),-1)

            #log center points
            points.append(center)
            print(center)
        
        #loop over the set of tracked points
        for i in range(1,len(points)):
                try:
                    cv2.line(frame,points[i-1],points[i],(0,255,0),2)
                except:
                    pass

        #make frame count zero
        frame_count = 0    
        out.write(frame)
    else:
        break

cap.release()
out.release()