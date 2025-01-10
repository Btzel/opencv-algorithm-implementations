import cv2
import numpy as np
import random
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
image = cv2.imread(os.path.join(script_dir, '../external/datasets/inpainting/images/IMG_7539.jpg'))
display(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#create histogram distribution
hist,bins = np.histogram(gray.flatten(),256,[0,256])

#get cumulative sum 
cdf = hist.cumsum()

#get normalize cumulative distribution
cdf_normalized = cdf * float(hist.max()) / cdf.max()

#plot cdf overlaid onto histogram
plt.plot(cdf_normalized,color = 'b')
plt.hist(gray.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc = 'upper left')
plt.show()

#equalize histogram
gray = cv2.equalizeHist(gray)
display(gray)

hist,bins = np.histogram(gray.flatten(),256,[0,256])

#get cumulative sum 
cdf = hist.cumsum()

#get normalize cumulative distribution
cdf_normalized = cdf * float(hist.max()) / cdf.max()

#plot cdf overlaid onto histogram
plt.plot(cdf_normalized,color = 'b')
plt.hist(gray.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc = 'upper left')
plt.show()
