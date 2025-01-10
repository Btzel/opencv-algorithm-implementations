import dlib
import cv2
import numpy as np
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

PREDICTOR_PATH = os.path.join(script_dir, '../external/datasets/random/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rect_array = []
    rects = detector(im,1)
    if len(rects) > 20:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    for i in range(0,len(rects)):
        rect_array.append(np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()]))

    return rect_array

def annotate_landmarks(im,landmarks):
    im = im.copy()
    for landmark in landmarks:
        for idx,point in enumerate(landmark):
            pos = (point[0,0],point[0,1])
            '''cv2.putText(im,str(idx),pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0,0,255))'''
            cv2.circle(im,pos,3,color=(0,255,255))

    return im

image = cv2.imread(os.path.join(script_dir, '../external/datasets/haarcascade/images/face/fbeda93731e1cfec.jpg'))
landmarks= get_landmarks(image)
print(landmarks)
image_with_landmarks= annotate_landmarks(image,landmarks)
display(image_with_landmarks)
        
