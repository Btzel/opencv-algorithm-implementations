import cv2
import numpy as np
import math
import os
import scipy.signal
from matplotlib import pyplot as plt


def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
# Load the image
script_dir = os.path.dirname(os.path.abspath(__file__))

image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/tilt_shift/original/cat017.jpeg'))

# Convert to HSV for color-based segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_orange = np.array([10, 100, 100])  # Hue, Saturation, Value (adjust as needed)
upper_orange = np.array([30, 255, 255])

# Define the range for white tones
lower_white = np.array([0, 0, 200])  # Bright regions with low saturation
upper_white = np.array([180, 50, 255])

# Define the range for dark orange tones
lower_dark_orange = np.array([5, 150, 50])  # Adjust for darker oranges
upper_dark_orange = np.array([15, 255, 150])

# Combine masks for light orange, dark orange, and white tones
orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
dark_orange_mask = cv2.inRange(hsv, lower_dark_orange, upper_dark_orange)
white_mask = cv2.inRange(hsv, lower_white, upper_white)

# Combine all masks
cat_mask = cv2.bitwise_or(orange_mask, dark_orange_mask)
cat_mask = cv2.bitwise_or(cat_mask, white_mask)

# Refine the combined mask
kernel = np.ones((5, 5), np.uint8)
cat_mask = cv2.morphologyEx(cat_mask, cv2.MORPH_CLOSE, kernel)
cat_mask = cv2.morphologyEx(cat_mask, cv2.MORPH_OPEN, kernel)


# Display or save the mask
cv2.imwrite(os.path.join(script_dir, '../external/datasets/random/tilt_shift/mask/cat017.jpeg'), cat_mask)
display(cat_mask)
