import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("E:/lly/tp/73216866e870e7610033edb1feeaf25.jpg")
orb = cv.ORB_create(nfeatures=500)
kp,des = orb.detectAndCompute(img,None)
print(des.shape)
img2 = cv.drawKeypoints(img,kp,None,color=(0,0,255),flags=0)
plt.figure(figsize=(10,8),dpi=100)
plt.imshow(img2[:,:,::-1])
plt.xticks([]),plt.yticks([])
plt.show()