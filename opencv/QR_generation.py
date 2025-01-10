import cv2
import numpy as np
from matplotlib import pyplot as plt
import qrcode
from PIL import Image
import qrcode.constants
import os
def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4
)

qr.add_data("https://github.com/Btzel")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save(os.path.join(script_dir, '../external/datasets/random/qrcode.png'))


qrcode = cv2.imread(os.path.join(script_dir, '../external/datasets/random/qrcode.png'))
display(qrcode)

