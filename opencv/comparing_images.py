import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def mse(image1,image2):
    #images must be of the same dimension
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error

def compare(image1,image2):
    image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    print("MSE: {:.2f}".format(mse(image1,image2)))
    print("SS: {:.2f}".format(structural_similarity(image1,image2)))

script_dir = os.path.dirname(os.path.abspath(__file__))

fireworks1 = cv2.imread(os.path.join(script_dir, '../external/datasets/random/fireworks1.png'))
fireworks2 = cv2.imread(os.path.join(script_dir, '../external/datasets/random/fireworks2.png'))

M =np.ones(fireworks1.shape,dtype="uint8") * 100
fireworks1b = cv2.add(fireworks1,M)

compare(fireworks1,fireworks1)
compare(fireworks1,fireworks1b)
compare(fireworks1,fireworks2)
compare(fireworks1b,fireworks2)

#can be used in motion detection

