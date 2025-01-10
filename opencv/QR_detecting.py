from pyzbar.pyzbar import decode
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
image = cv2.imread(os.path.join(script_dir, '../external/datasets/random/customqr.png'))

codes = decode(image)

for bc in codes:
    (x,y,w,h) = bc.rect

    pt1,pt2,pt3,pt4 = bc.polygon
    pts = np.array([[pt1.x,pt1.y],[pt2.x,pt2.y],[pt3.x,pt3.y],[pt4.x,pt4.y]],np.int32)

    pts = pts.reshape((-1,1,2))
    cv2.polylines(image,[pts],True,(0,0,255),3)

    barcode_text = bc.data.decode()
    barcode_type = bc.type

    text = "{} ({})".format(barcode_text,barcode_type)
    cv2.putText(image,barcode_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    cv2.putText(image,barcode_type,(x+w,y+h-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    print("QR Code revealed: {}".format(text))

display(image)

