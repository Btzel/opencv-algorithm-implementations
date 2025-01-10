import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def centroidHistogram(clt):
    #create a histogram for the clusters based on the pixels in each cluster
    #get the labels for each cluster
    numLabels = np.arange(0,len(np.unique(clt.labels_))+1)
    #create histogram
    (hist,_) = np.histogram(clt.labels_,bins=numLabels)
    #normalize the histogram, so that it sums to one 
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plotColors(hist,centroids):
    #create blank barchart
    bar = np.zeros((100,500,3),dtype="uint8")

    x_start = 0
    #iterate over the percentage and dominant color of each cluster
    for (percent,color) in zip(hist, centroids):
        #plot the relative percentage of each cluster
        end = x_start + (percent * 500)
        cv2.rectangle(bar,(int(x_start),0),(int(end),100),
                      color.astype("uint8").tolist(),-1)
        x_start = end
    return bar

script_dir = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(script_dir, '../external/datasets/animals/cat/cat001.jpg'))
display(image)

#reshape image into list of RGB pixels
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1],3))

number_of_clusters = 5

clt = KMeans(number_of_clusters)
clt.fit(image)

hist = centroidHistogram(clt)
bar = plotColors(hist,clt.cluster_centers_)

#show color bar
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


