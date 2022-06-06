# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:35:15 2022
@author: Harun Serkan Metin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
    
np.set_printoptions(threshold=np.inf)

image = cv2.imread("test1.jpg")
original = image 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = gray 

hist = cv2.calcHist([image],[0],None,[256],[0,255]) 
hist=hist.reshape(256,1)

y =np.cumsum(np.uint(hist))

minval = np.min(y[np.nonzero(y)])

y=((y-minval)/(image.size-minval))*255
y=np.round_(y)

s=image.shape

for i in range(s[0]):
    for j in range(s[1]):
        k=image[i][j]
        image[i][j]=y[k]
        
equalized = cv2.equalizeHist(gray)
extract=abs(image-equalized)

print("Not Matched Pixels: ",np.count_nonzero(extract))

#Show Images
images=[original,image,equalized,extract]
titles=["original","EqulizedMe","EqulizedCV2","Extracted"]
for x in range(len(images)):
    cv2.imshow(titles[x],images[x])
##################
#hist eq plot
"""
ih=cv2.calcHist([image],[0],None,[256],[0,256])
ie=cv2.calcHist([equalized],[0],None,[256],[0,256])
plt.plot(ie,label="OpenCV")
plt.plot(ih,label="Me")
plt.title("Histogram Equalization")
plt.legend()
plt.show()
"""

cv2.waitKey(0)
cv2.imwrite("eq_my.jpg",image)
cv2.imwrite("extracted.jpg",extract)


