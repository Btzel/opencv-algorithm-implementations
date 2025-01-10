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
image = cv2.imread(os.path.join(script_dir, '../external/datasets/inpainting/images/abraham.jpg'))
display(image)
marked_damages = cv2.imread(os.path.join(script_dir, '../external/datasets/inpainting/images/mask.jpg'),0)

ret, threshold = cv2.threshold(marked_damages,254,255,cv2.THRESH_BINARY)

#dilate (make thicher marks made)
#since thresholding has narrowed it slightly
kernel = np.ones((7,7),np.uint8)
mask = cv2.dilate(threshold,kernel,iterations=1)
cv2.imwrite(os.path.join(script_dir, '../external/datasets/inpainting/images/abraham_mask.png'),mask)
restored_img = cv2.inpaint(image,mask,3,cv2.INPAINT_TELEA)
display(restored_img)