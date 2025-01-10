import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile,join
from matplotlib import pyplot as plt

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))


#load t7 neural transfer models
model_file_path = os.path.join(script_dir, '../external/datasets/NeuralStyleTransfer/NeuralStyleTransfer/models/ECCV16/')
model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path,f))]

img= cv2.imread(os.path.join(script_dir, '../external/datasets/random/car.jpg'))

#loop through and applying each model style on input image
for (i,model) in enumerate(model_file_paths):
    #print the model is being used
    print(str(i+1) + ". Using model: " + str(model)[:-3])
    style = cv2.imread(os.path.join(script_dir, '../external/datasets/NeuralStyleTransfer/NeuralStyleTransfer/art/') + str(model)[:-3]+".jpg")
    #loading neural style model
    neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path+model)
    #resize to a fixed height of 640
    height,width = int(img.shape[0]),int(img.shape[1])
    newWidth = int((640/height) * width)
    resizedImg = cv2.resize(img,(newWidth,640),interpolation=cv2.INTER_AREA)
    #create blob from the image and then perform a forward pass run of the network
    inpBlob = cv2.dnn.blobFromImage(resizedImg,1.0,(newWidth,640),(103.939,116.779,123.68),swapRB=False,crop=False)

    neuralStyleModel.setInput(inpBlob)
    output = neuralStyleModel.forward()

    #reshape output tensor, adding back the mean subtraction and re ordering the channels
    output = output.reshape(3,output.shape[2],output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.68
    output /= 255
    output = output.transpose(1,2,0)

    #display original image, the style being applied and final neural style transfer
    display(img,"original")
    display(style,"style")
    display(output,"neural style transfer")