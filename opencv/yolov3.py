import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
from os import listdir
from os.path import isfile,join

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
labelsPath = os.path.join(script_dir, '../external/datasets/YOLO/YOLO/yolo/coco.names')
LABELS = open(labelsPath).read().strip().split("\n")
print(f"Number of labels: {len(LABELS)}")

COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")

print("Loading YOLO Weights..")

weightsPath = os.path.join(script_dir, '../external/datasets/YOLO/YOLO/yolo/yolov3.weights')
cfgPath = os.path.join(script_dir, '../external/datasets/YOLO/YOLO/yolo/yolov3.cfg')

#create our blob object
net = cv2.dnn.readNetFromDarknet(cfgPath,weightsPath)
#set our backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()

print("Starting detections")
mypath = os.path.join(script_dir, '../external/datasets/YOLO/YOLO/images/')
file_names = [f for f in listdir(mypath) if isfile(join(mypath,f))]

#loop through images run them through our classifier
for file in file_names:
    #load input image and grab spatial dimensions
    print(file)
    image = cv2.imread(mypath+file)
    (H,W) = image.shape[:2]

    #now we construct our blob from our input images
    blob = cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)
    #set input to image blob
    net.setInput(blob)
    #run a forward pass through the network
    layerOutputs = net.forward(ln)

    #initialize lists for detected bounding boxes, confidences and classes
    boxes = []
    confidences = []
    IDs = []

        # loop over each of the layer outputs
    for output in layerOutputs:
        print(f"Layer output shape: {output.shape}")  # Check shape of each layer output

        # loop over each detection in this output
        for detection in output:
            # Check if there are any detections
            if len(detection) < 85:
                continue  # Skip invalid detections (those without full information)

            scores = detection[5:]
            classID = np.argmax(scores)

            # Check if classID is within valid range
            if classID < len(LABELS):
                confidence = scores[classID]
                if confidence > 0.75:  # Only keep detections with high confidence
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    IDs.append(classID)
    
    #apply non-maxima suppression to reduce overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])

            #draw bounding boxes and put class label on the image
            color = [int(c) for c in COLORS[IDs[i]]]
            cv2.rectangle(image,(x,y),(x+w,y+h),color,3)
            text= "{}: {:.4f}".format(LABELS[IDs[i]],confidences[i])
            cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    display(image,"YOLO Detections",size=12)
    
