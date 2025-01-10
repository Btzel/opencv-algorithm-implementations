import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from barcode import EAN13
from barcode.writer import ImageWriter

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "../external/datasets/random/barcode.png")

with open(image_path,'wb') as f:
    EAN13('1928374650197',writer=ImageWriter()).write(f)

barcode = cv2.imread(image_path)
display(barcode)


