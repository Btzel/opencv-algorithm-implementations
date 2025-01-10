#<-- LIBRARIES -->
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_local

#<-- DISPLAY -->
script_dir = os.path.dirname(os.path.abspath(__file__))
#read
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
#rgb
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#gray_scale
###image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#getting scales
width,height = image.shape[0],image.shape[1]
#ratio
aspect_ratio=width/height
#create figure with the size of aspect_ratio and the scale you want
plt.figure(figsize=(10 * aspect_ratio,10))
#title
plt.title("mini mini bir kedi")
#imshow, cmap="gray" for gray_scale display
plt.imshow(image)
#show plot
#plt.show()

#<-- RGB COLOR SPACE -->

#read
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
#get each color space separately
B, G, R = cv2.split(image)
#create image matrix with image shape
color_space_matrix = np.zeros(image.shape[:2],dtype="uint8")
#fill with the single color space and display
red_channel=cv2.merge([color_space_matrix,color_space_matrix,R])
green_channel=cv2.merge([color_space_matrix,G,color_space_matrix])
blue_channel=cv2.merge([B,color_space_matrix,color_space_matrix])
#convert bgr to rgb and display
plt.imshow(cv2.cvtColor(blue_channel,cv2.COLOR_BGR2RGB))
#plt.show()
#merge the channels and display
plt.imshow(cv2.cvtColor(cv2.merge([B,G,R]),cv2.COLOR_BGR2RGB))
#plt.show()
#channel amplifying
plt.imshow(cv2.cvtColor(cv2.merge([B+100,G,R]),cv2.COLOR_BGR2RGB))
#plt.show()

#<-- HSV COLOR SPACE -->

#read
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
#convert to hsv (plt is designed for RGB only images so its not gonna be good visual)
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
plt.imshow(hsv_image)
#plt.show()
#hsv
hue = hsv_image[:,:,0]
saturation = hsv_image[:,:,1]
value = hsv_image[:,:,2]
plt.imshow(hue)
#plt.show()
plt.imshow(saturation)
#plt.show()
plt.imshow(value)
#plt.show()

#<-- Drawing images and shapes -->

#create empty matrix
image = np.zeros((512,512,3),np.uint8)
#draw whatever you want
cv2.line(image,(0,0),(511,511),(255,255,0),5)
cv2.rectangle(image,(100,100),(300,250),(255,255,255),5)
cv2.circle(image,(350,300),100,(255,0,255),1)
polygon_pts = np.array([[10,50],[400,50],[90,200],[50,500]],np.int32)
polygon_pts = polygon_pts.reshape((-1,1,2))
cv2.polylines(image,[polygon_pts],True,(0,255,255),3)
cv2.putText(image,"Hello burak",(200,300),cv2.FONT_HERSHEY_COMPLEX,1,(40,200,3),4)
plt.imshow(image)
#plt.show()

#<-- Transformations and rotations -->

#read
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
#define width and height
height,width = image.shape[:2]
#we will shift by quarter of the height and width
q_width,q_height = width/4,height/4
# our translation
#      | 1  0  Tx |
#  T = | 0  1  Ty |
# T is our translation matrix
T = np.float32([[1,0,q_width],[0,1,q_height]])
#we use warpAffine to transform the image using the matrix, T
image_translation = cv2.warpAffine(image,T,(width,height))
plt.imshow(cv2.cvtColor(image_translation,cv2.COLOR_BGR2RGB))
#plt.show()
#what does T looks like
#print(T)
#rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),90,1)
#rotating image
rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
plt.imshow(cv2.cvtColor(rotated_image,cv2.COLOR_BGR2RGB))
#plt.show()
#rotating image with transpose (flip on x and y)
transpose_image = cv2.transpose(image)
plt.imshow(cv2.cvtColor(transpose_image,cv2.COLOR_BGR2RGB))
#plt.show()
#flip image on axis (0 for y, 1 for x)
flipped_image = cv2.flip(image,1)
plt.imshow(cv2.cvtColor(flipped_image,cv2.COLOR_BGR2RGB))
#plt.show()

#<-- SCALING, RESIZING, INTERPOLATION and CROPPING -->

#read
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
#scale (reduce the size)
scaled_image = cv2.resize(image,None,fx=0.25,fy=0.25)
plt.imshow(cv2.cvtColor(scaled_image,cv2.COLOR_BGR2RGB))
#plt.show()
#scale (enlarge the size)
scaled_image = cv2.resize(image,None,fx=2,fy=2)
plt.imshow(cv2.cvtColor(scaled_image,cv2.COLOR_BGR2RGB))
#plt.show()
#scaling with inter cubic interpolation
scaled_image = cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(scaled_image,cv2.COLOR_BGR2RGB))
#plt.show()
#scaling with inter nearest interpolation
scaled_image = cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
plt.imshow(cv2.cvtColor(scaled_image,cv2.COLOR_BGR2RGB))
#plt.show()
#scale with skewing the resizing by setting exact dimensions
scaled_image = cv2.resize(image,(900,400),interpolation=cv2.INTER_AREA)
plt.imshow(cv2.cvtColor(scaled_image,cv2.COLOR_BGR2RGB))
#plt.show()
#Image pyramids (resize by *2 or /2 as default)
smaller_image = cv2.pyrDown(image)
plt.imshow(cv2.cvtColor(smaller_image,cv2.COLOR_BGR2RGB))
#plt.show()
larger_image = cv2.pyrUp(image)
plt.imshow(cv2.cvtColor(larger_image,cv2.COLOR_BGR2RGB))
#plt.show()
#cropping
height, width = image.shape[:2]
starting_row,starting_col = int(height * .25), int(width * .25)
ending_row,ending_col = int(height * .75), int(width * .75)
cropped_image = image[starting_row:ending_row,starting_col:ending_col]
plt.imshow(cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB))
#plt.show()

#<-- Arithmetic Operations -->

#read image as gray_scaled
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'),0)
#create matrix of ones, then multiply it by a scaler of 100
M = np.ones(image.shape,dtype="uint8") * 100
#now if we add it to the image, its brightness will increase (note: for example if you add to 100 to a pixel with value 200 directly by image+M,
#it will reset after 255 so it will be 45 so it is gonna be black so use np.add to avoid it)
brighter_image = cv2.add(image,M)
plt.imshow(cv2.cvtColor(brighter_image,cv2.COLOR_BGR2RGB))
#plt.show()
#darker image
darker_image = cv2.subtract(image,M)
plt.imshow(cv2.cvtColor(darker_image,cv2.COLOR_BGR2RGB))
#plt.show()

#<-- Bitwise Operations and Masking -->

#creating square and ellipse
square = np.zeros((300,300),np.uint8)
ellipse = np.zeros((300,300),np.uint8)
cv2.rectangle(square, (50,50),(250,250),255,-2)
cv2.ellipse(ellipse,(150,150),(150,150),30,0,180,255,-1)
#shows only where they intersect (intersection)
bitwiseAnd_areas = cv2.bitwise_and(square, ellipse)
plt.imshow(cv2.cvtColor(bitwiseAnd_areas,cv2.COLOR_BGR2RGB))
#plt.show()
#shows where either square or ellipse (union)
bitwiseOr_areas = cv2.bitwise_or(square, ellipse)
plt.imshow(cv2.cvtColor(bitwiseOr_areas,cv2.COLOR_BGR2RGB))
#plt.show()
#shows where either exists by itself
bitwiseXor_areas = cv2.bitwise_xor(square,ellipse)
plt.imshow(cv2.cvtColor(bitwiseXor_areas,cv2.COLOR_BGR2RGB))
#plt.show()
#shows everything that isnt part of the square (so the white parts is shown as not square)
bitwiseNot_square = cv2.bitwise_not(square)
plt.imshow(cv2.cvtColor(bitwiseNot_square,cv2.COLOR_BGR2RGB))
#plt.show()

#<-- Convolutions, Blurring and Sharpening images -->

#read
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
#creating 3x3 kernel
kernel_3x3 = np.ones((3,3),np.float32) / 9
#blurring using convolution using cv2.filter2D to convolve the kernel with an image
blurred_image = cv2.filter2D(image,-1,kernel_3x3)
plt.imshow(cv2.cvtColor(blurred_image,cv2.COLOR_BGR2RGB))
#plt.show()
#with bigger kernel
kernel_7x7 = np.ones((7,7),np.float32)/49
blurred_image = cv2.filter2D(image,-1,kernel_7x7)
plt.imshow(cv2.cvtColor(blurred_image,cv2.COLOR_BGR2RGB))
#plt.show()
#blurring with cv2.blur avering the box and replacin the center element in it content
blurred_image = cv2.blur(image,(5,5))
plt.imshow(cv2.cvtColor(blurred_image,cv2.COLOR_BGR2RGB))
#plt.show()
#gaussian blur uses gaussian kernel instead of box
blurred_image = cv2.GaussianBlur(image,(5,5),0)
plt.imshow(cv2.cvtColor(blurred_image,cv2.COLOR_BGR2RGB))
#plt.show()
#median blur takes median of all pixels under kernel area and central element is replaced with this median value
blurred_image = cv2.medianBlur(image,5)
plt.imshow(cv2.cvtColor(blurred_image,cv2.COLOR_BGR2RGB))
#plt.show()
#Bilateral blurring is very effective in noise removal while keeping edges sharp (it is like focusing and blurring others but not totaly)
bilateral_image = cv2.bilateralFilter(image,9,75,75)
plt.imshow(cv2.cvtColor(bilateral_image,cv2.COLOR_BGR2RGB))
#plt.show()
#denoising
denoised_image = cv2.fastNlMeansDenoisingColored(image,None,6,6,7,21)
plt.imshow(cv2.cvtColor(denoised_image,cv2.COLOR_BGR2RGB))
#plt.show()
#sharpening image with kernel
sharpening_kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

sharpened_image = cv2.filter2D(image,-1,sharpening_kernel)
plt.imshow(cv2.cvtColor(sharpened_image,cv2.COLOR_BGR2RGB))
#plt.show()

#<-- Thresholding, Binarization and Adaptive Thresholding -->

#read image as gray_scale
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'),0)
#values below 127 goes to 0 or black, everything above goes to 255 (white)
ret,threshold_image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#value below 127 goes to 255 white, above goes to 0 (black)
ret,threshold_image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#value above 127 are truncated (held at 127) (the 255 argument is unused)
ret,threshold_image = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#value below 127 are go to 0 above 127 is unchange
ret,threshold_image = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#reverse of above, below  127 is unchanged above goes to 0
ret,threshold_image = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#adaptive mean thresholding using cv2.adaptiveThreshold
threshold_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,5)
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#Otsu's thresholding after gaussian blur
blurred_image = cv2.GaussianBlur(image,(5,5),0)
_, threshold_image = cv2.threshold(blurred_image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#skimage threshold local
#read image
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
V = cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V,25,offset=15,method='gaussian')
threshold_image = (V > T).astype("uint8") * 255
plt.imshow(cv2.cvtColor(threshold_image,cv2.COLOR_BGR2RGB))
#plt.show()
#####!!! blurring is important before thresholding..

#<-- Dilation, Erosion, and Edge Detection -->

#read image as gray_scale
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'),0)
#define kernel
kernel = np.ones((5,5),np.uint8)
#erosion
eroded_image = cv2.erode(image,kernel,iterations=1)
plt.imshow(cv2.cvtColor(eroded_image,cv2.COLOR_BGR2RGB))
#plt.show()
#dilation
dilated_image = cv2.dilate(image,kernel,iterations=1)
plt.imshow(cv2.cvtColor(dilated_image,cv2.COLOR_BGR2RGB))
#plt.show()
#Opening - good for removing noise
opened_image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
plt.imshow(cv2.cvtColor(opened_image,cv2.COLOR_BGR2RGB))
#plt.show()
#Closing - good for removing noise
closed_image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
plt.imshow(cv2.cvtColor(closed_image,cv2.COLOR_BGR2RGB))
#plt.show()
#Edge detection needs a threshold to tell what difference/change should be counted as an edge 
#the first threshold gradient
canny_image = cv2.Canny(image, 50, 120)
plt.imshow(cv2.cvtColor(canny_image,cv2.COLOR_BGR2RGB))
#plt.show()
#wide edge thresholds expect lots of edges
canny_image = cv2.Canny(image,10,200)
plt.imshow(cv2.cvtColor(canny_image,cv2.COLOR_BGR2RGB))
#plt.show()
#Narrow threshold, expect less edges
canny_image = cv2.Canny(image,200,240)
plt.imshow(cv2.cvtColor(canny_image,cv2.COLOR_BGR2RGB))
#plt.show()
canny_image = cv2.Canny(image,60,110)
plt.imshow(cv2.cvtColor(canny_image,cv2.COLOR_BGR2RGB))
#plt.show()

# we provide two values as threshold1 and threshold2. any gradient value larger than threshold2
# is considered to be an edge. Any value below threshold1 is considered not to be an edge.
# Values in the between two thresholds are either classified as edges or non-edges based on how
# their intensities are "connected". In this case, any gradient values below 60 are considered
# non-edges whereas any values above 120(for the first canny example) are considered as edges

#a function that finds an optimal thresholds based on median image pixel intensity
def autoCanny(image):
    blurred_image = cv2.blur(image,ksize=(5,5))
    median_value = np.median(image)
    lower = int(max(0,0.66 * median_value))
    upper = int(min(255,1.33 * median_value))
    edges = cv2.Canny(image,lower,upper)
    return edges
autoCanny_image = autoCanny(image)
plt.imshow(cv2.cvtColor(autoCanny_image,cv2.COLOR_BGR2RGB))
#plt.show()
