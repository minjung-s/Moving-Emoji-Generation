import dlib
import cv2
import numpy as np
import json
import argparse

import cv2
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt # used to plot our images
from pylab import rcParams # used to change image size

def landmark_detect(img_path, img_save_name):

    # create list for landmarks
    ALL = list(range(0, 68))
    JAWLINE = list(range(0, 17))
    RIGHT_EYEBROW = list(range(17, 22))
    LEFT_EYEBROW = list(range(22, 27))
    RIGHT_EYE = list(range(36, 42))
    LEFT_EYE = list(range(42, 48))
    NOSE = list(range(27, 36))
    MOUTH_OUTLINE = list(range(48, 61))
    MOUTH_INNER = list(range(61, 68))

    # create face detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    final=[]

    img = dlib.load_rgb_image(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = detector(img, 1)

    landmarks = predictor(img, face_detector[0])  #68개 점 찾기
    
    #create list to contain landmarks
    landmark_list = []

    # append (x, y) in landmark_list
    for p in landmarks.parts():
        landmark_list.append([p.x, p.y])
        cv2.circle(img, (p.x, p.y), 2, (0, 255, 0), -1)

    key_val = [ALL, landmark_list]
    landmark_dict = dict(zip(*key_val))
    final.append(landmark_dict)

    with open(img_save_name, "w") as json_file:
        json_file.write(json.dumps(final))
        json_file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path" ,required= False, default = "image", help="image path")
    parser.add_argument("--save_path" ,required= False, default = "result", help="save image path")
    args = parser.parse_args()

    image_path = args.image_path
    img_list = os.listdir(image_path)
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if len(img_list)==0:
        assert False, "no images"

    for img_name in img_list:   

        image_path = args.image_path

        image_path =  os.path.join(image_path,img_name)
        temp = img_name.split('.') # 저장할 image name
        img_save_name = temp[0] + '.json'
        all_save_path = os.path.join(save_path, img_save_name)

        landmark_detect(image_path, all_save_path)
