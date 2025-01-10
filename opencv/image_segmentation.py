#<-- Libraries -->
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
#contours
#read image
script_dir = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/shapes.jpeg'))
#convert to gray_scale
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#threshold
_,threshold_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(threshold_image)
#plt.show()
#finding contours
contours, hierarchy = cv2.findContours(threshold_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#draw all contours, note this overwrites the input image (inplace operation)
#use '-1' as the 3rd parameter to draw all
cv2.drawContours(image,contours,-1,(0,255,0),thickness=2)
plt.imshow(image)
#plt.show()
#number of contours found
#print("number of contours found: " +  str(len(contours))) 
#using canny edge detection instead thresholding
canny_image = cv2.Canny(gray_image,200,240)
#finding contours
contours, hierarchy = cv2.findContours(canny_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#draw all contours
cv2.drawContours(image,contours,-1,(0,255,0),thickness=2)
plt.imshow(image)
#plt.show()
#number of contours found
#print("number of contours found: " +  str(len(contours)))

#while doing contouring remember to using gray_scale image, and threshold or canny edge detection before contouring to binarize image
#Note: blurring before thresholding or canny edge is recommended to remove noisy contours

#finding contours
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/shapes.jpeg'))
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray_image,250,255)
contours, hierarchy = cv2.findContours(canny_image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contours = [c for c in contours if cv2.moments(c)['m00'] != 0]
cv2.drawContours(image,contours,-1,(0,255,0),thickness=2)
plt.imshow(image)
#plt.show()
#sorting area using cv2.ContourArea and cv2.Moments
def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

#sort contours by area in descending order
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#filter contours where the area (M['m00']) is not zero

#iterate over our contours and draw
for (i, c) in enumerate(sorted_contours):
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # Annotate the image with the contour number
    cv2.putText(image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.3, (0, 255, 0), 1)
    # Draw the contour
    cv2.drawContours(image, [c], -1, (255, 0, 0), 3)
# Display the image
plt.imshow(image)
#plt.show()

#another approximation but numeration is done by left to right depending on x axis so the most left contour area is the first
def x_cord_contour(contour):
    """Returns the x-coordinate of a contour's centroid for sorting."""
    M = cv2.moments(contour)
    if M['m00'] > 0:  # Avoid division by zero
        return int(M['m10'] / M['m00'])
    return float('inf')

def label_contour_center(image,c):
    #places red circle on the centers of contours
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #draw the contour number on the image
    cv2.circle(image,(cx,cy),1,(0,0,255),-1)
    return image

#read image
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/shapes.jpeg'))
original_image = image.copy()

for (i,c) in enumerate(contours):
    orig=label_contour_center(image,c)
plt.imshow(image)
#plt.show()
#sorting contours left to right
contours_left_to_right = sorted(contours,key=x_cord_contour,reverse=False)
#labeling contours left to right
for(i,c) in enumerate(contours_left_to_right):
    cv2.drawContours(original_image,[c],-1,(0,0,255),3)
    M=cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.putText(original_image,str(i+1),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1)
    (x,y,w,h)=cv2.boundingRect(c)
plt.imshow(original_image)
#plt.show()


# approximating contours using ApproxPolyDP
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/shapes.jpeg'))
original_image = image.copy()
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,threshold_image=cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
contours,hierarchy=cv2.findContours(threshold_image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
copy_image = image.copy()
for c in contours:
    x,y,w,h=cv2.boundingRect(c)
    cv2.rectangle(original_image,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.drawContours(image,[c],0,(0,255,0),1)
plt.imshow(image)
#plt.show()
plt.imshow(original_image)
#plt.show()
#iterate through each contour and compute the approx contour
for c in contours:
    #calculate accuracy as a percent of the contour perimeter
    accuracy = 0.03 * cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(copy_image,[approx],0,(0,255,0),1)
plt.imshow(copy_image)
#plt.show()

#convex hull
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/shapes.jpeg'))
original_image = image.copy()
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,threshold_image = cv2.threshold(gray_image,176,255,0)
contours,hierarchy=cv2.findContours(threshold_image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
for c in contours:
    cv2.drawContours(image,[c],0,(0,255,0),1)
plt.imshow(image)
#plt.show()
n=len(contours)-1
contours = sorted(contours,key=cv2.contourArea,reverse=False)[:n]
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(original_image,[hull],0,(0,255,0),1)
plt.imshow(original_image)
plt.show()

#matching contours
shape = cv2.imread(os.path.join(script_dir, '../external/datasets/random/shape.jpeg'),0)
shapes = cv2.imread(os.path.join(script_dir, '../external/datasets/random/shapes.jpeg'))
shapes = cv2.cvtColor(shapes,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(shape,127,255,0)
ret,thresh2 = cv2.threshold(shapes,127,255,0)

contours,hierarchy=cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
template_contour = contours[1]

contours,hierarchy=cv2.findContours(thresh2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    #iterate through each contour in the target image(shapes) and
    #use cv2.matchShapes to compare contour shapes
    match=cv2.matchShapes(template_contour,c,3,0.0)
    #print(match)
    #if the match value is less than 0.15
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []
    

cv2.drawContours(shape,[template_contour],-1,(0,255,0),3)
plt.imshow(shape)
#plt.show()

cv2.drawContours(shapes,[closest_contour],-1,(0,255,0),3)
plt.imshow(shapes)
#plt.show()

#not always working the best, it chose the true one with match value 0.03, but there is another with 0.01 match value


#Line detection using hough lines
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/line.jpeg'))
plt.imshow(image)
#plt.show()

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image,100,170,apertureSize=3)
plt.imshow(edges)
#plt.show()

#run houghlines using a rho accuracy of 1 pixel
#theta accuracy of np.pi / 180 which is 1 degree
#our line threshold is set to 240 (number of the points on the line)
lines = cv2.HoughLines(edges,1,np.pi/180,240)

#then we iterate through each line and convert it to the format
#required by cv2.lines (i.e. requiring end points)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),1)
plt.imshow(image)
#plt.show()

#Probabilistic hough lines
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/line.jpeg'))
plt.imshow(image)
#plt.show()

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image,100,170,apertureSize=3)
plt.imshow(edges)
#plt.show()

#again we use the same rho and theta accuracies
#however we specify a minimum vote (pts along line) of 100
#and min line length of 3 pixels and map gap between lines of 25 pixels
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,3,25)
#print(lines.shape)

for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),1)

plt.imshow(image)
#plt.show()

#hough circle detection
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/circle.jpeg'))
plt.imshow(image)
#plt.show()

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray_image,5)
circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1.2,25)

cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT,1.2,100)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    #draw outer circle
    cv2.circle(image,(i[0],i[1]),i[2],(0,0,255),5)

    #draw the center of the circle
    cv2.circle(image,(i[0],i[1]),2,(0,0,255),8)

plt.imshow(image)
#plt.show()

#blob detection
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/blob.jpg'),cv2.IMREAD_GRAYSCALE)
#set up the detector with default params
detector = cv2.SimpleBlobDetector.create()
#detect blobs
keypoints = detector.detect(image)
#draw detected keypoints as red circles
#cv2.DRAW_MATCHES_FLAGS_RICH_KEYPOINTS ensures the size of
#the circle corresponds to the size of blob
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image,keypoints,blank,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(blobs)
#plt.show()

#detecting circular blobs
params = cv2.SimpleBlobDetector.Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.9
params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector.create(params)
keypoints = detector.detect(image)
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image,keypoints,blank,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.title("blobs with custom parameters to find circles")
plt.imshow(blobs)
#plt.show()


#finding waldo using template matching
waldo = cv2.imread(os.path.join(script_dir, '../external/datasets/random/waldo.jpg'),0)
waldo_beach = cv2.imread(os.path.join(script_dir, '../external/datasets/random/waldobeach.jpg'))
gray_waldo_beach = cv2.cvtColor(waldo_beach,cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(gray_waldo_beach,waldo,cv2.TM_CCOEFF)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
#print(min_val,max_val,max_loc,min_loc)
#create bounding box
top_left = max_loc
bottom_right = (top_left[0] + 50,top_left[1]+50)
cv2.rectangle(waldo_beach,top_left,bottom_right,(0,0,255),5)
waldo_beach = cv2.cvtColor(waldo_beach,cv2.COLOR_BGR2RGB)
plt.imshow(waldo_beach)
plt.show()


#corner detection using harris corner detection
chess_board = cv2.imread(os.path.join(script_dir, '../external/datasets/random/chess.jpeg'))
gray = cv2.cvtColor(chess_board,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
harris_corners = cv2.cornerHarris(gray,3,3,0.05)
kernel = np.ones((7,7),np.uint8)
harris_corners = cv2.dilate(harris_corners,kernel,iterations=2)
chess_board[harris_corners > 0.6 * harris_corners.max()] = [255,127,127]
plt.imshow(chess_board)
#plt.show()

#corner detection with cv2.goodFeaturesToTrack
chess_board = cv2.imread(os.path.join(script_dir, '../external/datasets/random/chess.jpeg'))
gray = cv2.cvtColor(chess_board,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,80,0.0005,10)
for corner in corners:
    x,y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(chess_board,(x-10,y-10),(x+10,y+10),(0,255,0),2)
plt.imshow(chess_board)
plt.show()

