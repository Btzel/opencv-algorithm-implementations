import cv2
import os
from matplotlib import pyplot as plt

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))

image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/car.jpg'))

blur_1 = cv2.GaussianBlur(image,(5,5),0)
blur_2 = cv2.GaussianBlur(image,(9,9),0)
blur_3 = cv2.GaussianBlur(image,(13,13),0)

def getBlurScore(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image,cv2.CV_64F).var()

print("Blur Score: {}".format(getBlurScore(image)))
print("Blur Score: {}".format(getBlurScore(blur_1)))
print("Blur Score: {}".format(getBlurScore(blur_2)))
print("Blur Score: {}".format(getBlurScore(blur_3)))