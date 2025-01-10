import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import face_recognition

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))


known_image = face_recognition.load_image_file(os.path.join(script_dir, '../external/datasets/inpainting/images/Trump.jpg'))
unknown_image = face_recognition.load_image_file(os.path.join(script_dir, '../external/datasets/inpainting/images/Truiumph.png'))

unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
trump_encoding = face_recognition.face_encodings(known_image)[0]

result = face_recognition.compare_faces([trump_encoding],unknown_encoding)
print(f'Face matching is {result[0]}')