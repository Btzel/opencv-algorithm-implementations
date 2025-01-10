import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
def display(title="Image",image=None,size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
#we will use perspective transform to fix the perspective of the contoured object
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/perspective1.jpeg'))
display(image=image)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image,contours,-1,(0,255,0),1)
display(image=image)
print("number of contours found: " ,str(len(contours)))
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
for cnt in sorted_contours:
    #approximate contour
    perimeter = cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,0.05*perimeter,True)

    if len(approx) == 4:
        break

approx_points = approx.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")
s = approx_points.sum(axis=1)
rect[0] = approx_points[np.argmin(s)]
rect[2] = approx_points[np.argmax(s)]
diff = np.diff(approx_points, axis=1)
rect[1] = approx_points[np.argmin(diff)]
rect[3] = approx_points[np.argmax(diff)]

# Compute width and height of the new image
widthA = np.linalg.norm(rect[2] - rect[3])  # Bottom width
widthB = np.linalg.norm(rect[1] - rect[0])  # Top width
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(rect[1] - rect[2])  # Right height
heightB = np.linalg.norm(rect[0] - rect[3])  # Left height
maxHeight = int(max(heightA, heightB))

inputPts = rect
outputPts = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")
#get transformation matrix M
M = cv2.getPerspectiveTransform(inputPts,outputPts)
#apply the transform matrix M using warp perspective
dst = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
display("proper image", dst)


