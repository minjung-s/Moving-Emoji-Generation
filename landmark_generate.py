import dlib
import cv2
import torch
import numpy as np
import json
import argparse
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt
from pylab import rcParams
from PIL import Image
from utils import videos_to_numpy, save_videos

def landmark_detect(img_path):

    # create list for landmarks
    ALL = list(range(0, 68))

    # create face detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('params/shape_predictor_68_face_landmarks.dat')

    final=[]

    img = dlib.load_rgb_image(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = detector(img, 1)

    landmarks = predictor(img, face_detector[0])  
    #create list to contain landmarks
    landmark_list = []

    # append (x, y) in landmark_list
    for p in landmarks.parts():
        landmark_list.append([p.x, p.y])
        cv2.circle(img, (p.x, p.y), 2, (0, 255, 0), -1)

    key_val = [ALL, landmark_list]
    landmark_dict = dict(zip(*key_val))
    final.append(landmark_dict)

    return final

def sol1_generate_landmark(landmark, condition):
    landmark = landmark[0]
    input_landmark = []

    for I in range(68):
        temp = landmark[I]
        input_landmark.extend(temp)

    input_landmark = np.array(input_landmark)

    cond_dict = {'disgusted':0, 'happiness':1, 'surprised':2}
    c = cond_dict[condition]

    input_landmark = torch.tensor(np.exp(input_landmark/1024))
    input_landmark = input_landmark.view(1,136)
    input_landmark = input_landmark.type(torch.FloatTensor)


    c = torch.tensor(c)
    c = c.type(torch.FloatTensor)

    generator = torch.load('params/generator_exp.pytorch')
    with torch.no_grad():
        input_landmark = input_landmark.cuda()
        c = c.cuda()
        outputs = generator(input_landmark,c)

    outputs = outputs.view(50, -1) 
    outputs = outputs.detach().cpu().numpy()
    outputs = np.log(outputs)
    outputs = outputs * 1024

    output_landmark = dict()
    for t in range(50):
        t_landmark = outputs[t]
        temp_dict = dict()
        for i in range(68):
            if i < 17:
                temp_dict[str(i)] = landmark[i]
            else:
                temp_dict[str(i)] = list(map(lambda x: int(round(x)),t_landmark[2*i:2*(i +1)]))

        output_landmark[str(t)] = temp_dict


    return output_landmark