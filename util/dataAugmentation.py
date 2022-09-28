import numpy as np
from PIL import Image
import cv2 as cv
import random

def ranrange(a,b):
    return a+(b-a)*random.random()


def rotate(img,angle):
    dimension=img.shape[:2]
    rotpoint=[s//2 for s in dimension]
    matrix=cv.getRotationMatrix2D(rotpoint,angle,1.0)
    return cv.warpAffine(img,matrix,dimension)

def scale(img,factor):  
    xoffset=(img.shape[0]-img.shape[0]*factor)*0.5
    mat=np.asarray([[factor,0,xoffset],[0,factor,xoffset]])
    return cv.warpAffine(img,mat,img.shape)

def translate(img,xoffset,yoffset):  
    mat=np.asarray([[1,0,xoffset],[0,1,yoffset]])
    return cv.warpAffine(img,mat,img.shape)#getRotationMatrix2D(rotpoint,angle,1.0)

def augment(img):
    if random.random()<0.5:
        angle=ranrange(-60,60)
        img=rotate(img,angle)
    if random.random()<0.5:
        factor=ranrange(0.5,1.2)
        img=scale(img,factor)
    if random.random()<0.5:
        xoff,yoff=[2*(random.random()-0.5)*5 for i in range(2)]
        img=translate(img,xoff,yoff)
    if random.random()<0.5:
        img+=ranrange(0,0.1)*np.random.rand(*img.shape)
    return img