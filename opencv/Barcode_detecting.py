from pyzbar.pyzbar import decode
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
image_path = os.path.join(script_dir, "../external/datasets/random/barcode.png")
image = cv2.imread(image_path)



barcodes = decode(image)

for bc in barcodes:
    (x,y,w,h) = bc.rect

    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)


    barcode_text = bc.data.decode()
    barcode_type = bc.type

    text = "{} ({})".format(barcode_text,barcode_type)
    cv2.putText(image,barcode_text,(x-40,y+100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,barcode_type,(x+w,y+h-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    print("QR Code revealed: {}".format(text))

display(image)

