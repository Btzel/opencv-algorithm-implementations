import cv2
import numpy as np
import os
import random
from matplotlib import pyplot as plt

def display(title="Image",image=None,size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

#haar cascade is used to classify object, it can be evaluated as object detection
script_dir = os.path.dirname(os.path.abspath(__file__))
#Simple eye and face detection using haar cascade classifiers
path = os.path.join(script_dir, '../external/datasets/haarcascade/images/face')
all_files = os.listdir(path)
image_files = [file for file in all_files if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
random_image = random.choice(image_files)
image_path = os.path.join(path, random_image)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#we create our classifier with the xml file which is created for face detection parameters
face_classifier = cv2.CascadeClassifier(os.path.join(script_dir, '../external/datasets/haarcascade/xml/haarcascade_frontalface_default.xml'))
eye_classifier = cv2.CascadeClassifier(os.path.join(script_dir, '../external/datasets/haarcascade/xml/haarcascade_eye.xml'))
#classifier returns the region of interest of the detected face as a tuple (x,y,w,h)
#it stores the top left coordinate as x y and gives the width and height, so we can calculate the area of detection
faces = face_classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3)
#when no faces detected, classifier returns an empty tuple
if len(faces) == 0:
    print("No faces in this image")
else:
    #we iterate through our faces array and draw a rectangle
    #over each face in faces
    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),1)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = image[y:y+h,x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray,scaleFactor=1.2,minNeighbors=3)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),1)
display("eyes and faces in the image",image)


#Simple pedestrian (human body) detection using haar cascade classifiers
path = os.path.join(script_dir, '../external/datasets/haarcascade/images/pedestrian')
all_files = os.listdir(path)
image_files = [file for file in all_files if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
random_image = random.choice(image_files)
image_path = os.path.join(path, random_image)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#we create our classifier with the xml file which is created for pedestrian detection parameters
body_classifier = cv2.CascadeClassifier(os.path.join(script_dir, '../external/datasets/haarcascade/xml/haarcascade_fullbody.xml'))
bodies = body_classifier.detectMultiScale(gray)
if len(bodies) == 0:
    print("no pedestrian found there")
else:
    #we iterate through our faces array and draw a rectangle of region of interest
    for(x,y,w,h) in bodies:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
    display("pedestrian detection via hcc",image)

#not working good for my pedestrian dataset..



#Simple car detection using haar cascade classifiers
path = os.path.join(script_dir, '../external/datasets/haarcascade/images/car')
all_files = os.listdir(path)
image_files = [file for file in all_files if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
random_image = random.choice(image_files)
image_path = os.path.join(path, random_image)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#we create our classifier with the xml file which is created for car detection parameters
car_classifier = cv2.CascadeClassifier(os.path.join(script_dir, '../external/datasets/haarcascade/xml/haarcascade_cars.xml'))
cars = car_classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3)
if len(cars) == 0:
    print("no car in found in the image")
else:
    for(x,y,w,h) in cars:
        cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),1)
    display("cars detections in the image",image)

#detects cars and others things as well, hard code detection is not a good way to make object detection in 2025 :D



    





 




