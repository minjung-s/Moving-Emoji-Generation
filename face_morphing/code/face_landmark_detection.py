import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2
from imutils import face_utils

class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass

def generate_face_correspondences(theImage1, landmark2):
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('code/utils/shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68,2))
    
    img1 = theImage1
    list1 = []
    size = (img1.shape[0],img1.shape[1])
    currList = list1
    

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.

    dets = detector(img1, 1)

    try:
        if len(dets) == 0:
            raise NoFaceFound
    except NoFaceFound:
        print("Sorry, but I couldn't find a face in the image.")

    # j=j+1

    for k, rect in enumerate(dets):
        
        # Get the landmarks/parts for the face in rect.
        shape = predictor(img1, rect)
        # corresp = face_utils.shape_to_np(shape)
        
        for i in range(0,68):
            x = shape.part(i).x
            y = shape.part(i).y
            currList.append((x, y))
            corresp[i][0] += x
            corresp[i][1] += y
            # cv2.circle(img1, (x, y), 2, (0, 255, 0), 2)

        # Add back the background
        currList.append((1,1))
        currList.append((size[1]-1,1))
        currList.append(((size[1]-1)//2,1))
        currList.append((1,size[0]-1))
        currList.append((1,(size[0]-1)//2))
        currList.append(((size[1]-1)//2,size[0]-1))
        currList.append((size[1]-1,size[0]-1))
        currList.append(((size[1]-1),(size[0]-1)//2))


    list2 = landmark2
    # print(corresp)
    # exit()
    for i in range(0,68):
        corresp[i][0] += list2[i][0]
        corresp[i][1] += list2[i][1]

    list2.append((1,1))
    list2.append((size[1]-1,1))
    list2.append(((size[1]-1)//2,1))
    list2.append((1,size[0]-1))
    list2.append((1,(size[0]-1)//2))
    list2.append(((size[1]-1)//2,size[0]-1))
    list2.append((size[1]-1,size[0]-1))
    list2.append(((size[1]-1),(size[0]-1)//2))

    
    # Add back the background
    narray = corresp/2
    narray = np.append(narray,[[1,1]],axis=0)
    narray = np.append(narray,[[size[1]-1,1]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = np.append(narray,[[1,size[0]-1]],axis=0)
    narray = np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = np.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)

    
    return [size,img1,list1,list2,narray]
