import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def addWhiteNoise(image):
    prob = random.uniform(0.05,0.1)
    
    rnd = np.random.rand(image.shape[0],image.shape[1])

    image[rnd < prob] = np.random.randint(50,230)
    return image

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "../external/datasets/inpainting/images/londonxmas.jpeg")
image = cv2.imread(image_path)

display(image)

#apply white noise 
noise_1 = addWhiteNoise(image)
display(noise_1)

#cv2.fastNlMeansDenoisingColored
#None are - the filter strength 'h' (5-12 is a good range)
#next is hForColorComponents, set as same value as h again
#templateWindowSize (odd numbers only) rec. 7
#searchWindowSize (odd numbers only) rec. 21

dst = cv2.fastNlMeansDenoisingColored(noise_1,None,11,6,7,21)
display(dst)
