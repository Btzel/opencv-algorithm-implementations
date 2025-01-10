import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(script_dir, '../external/datasets/random/OCR1.png'))

output_txt = pytesseract.image_to_string(img)

print("PyTeserract Extracted: {}".format(output_txt))

#thresholding makes it work so much better before extracting texts