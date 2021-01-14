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

    return final

    # with open(img_save_name, "w") as json_file:
    #     json_file.write(json.dumps(final))
    #     json_file.write('\n')

def sol1_generate_landmark(landmark, condition):
    # landmark = landmark[0]
    input_landmark = []

    for i in range(68):
        temp = landmark[str(i)]
        input_landmark.extend(temp)

    input_landmark = np.array(input_landmark)

    cond_dict = {'disgust':0, 'happiness':1, 'surprise':2}
    c = cond_dict[condition]

    input_landmark = torch.tensor(np.exp(input_landmark/1024))
    input_landmark = input_landmark.view(1,136)
    input_landmark = input_landmark.type(torch.FloatTensor)

    c = torch.tensor(c)
    c = c.type(torch.FloatTensor)

    generator = torch.load('generator_exp.pytorch')
    with torch.no_grad(): 
        input_landmark = input_landmark.cuda()
        c = c.cuda()
        outputs = generator(input_landmark,c)

    outputs = outputs.view(50, -1) # 50시점
    outputs = outputs.detach().numpy()
    outputs = np.log(outputs)
    outputs = outputs * 1024

    landmark = dict()
    for t in range(50):
        t_landmark = outputs[t]
        temp_dict = dict()
        for i in range(68):
            if i < 17: #1-17 landmark는 source landmark의 좌표값 유지
                temp_dict[str(i)] = data[str(i)]
            else:
                temp_dict[str(i)] = list(map(lambda x: int(round(x)),t_landmark[2*i:2*(i +1)]))

        landmark[str(t)] = temp_dict

    return landmark


def sol2_generate_landmark(file_path, output_type, condition, duration):
    img = Image.open(file_path).resize((64,64), Image.LANCZOS)
    img = np.array(img)
    img = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).cuda()

    condition_dic = {'disgusted': [1,0,0], 'happiness': [0,1,0], 'surprise': [0,0,1]}
    target_class = torch.tensor(condition_dic[condition])

    target_class = target_class.repeat(16,1).cuda()


    generator = torch.load(--???--).cuda()
    with torch.no_grad(): 
        v, _ = generator.sample_videos(img, 1, target_class, 16)
        video = videos_to_numpy(v)

    saved_name = file_path[:-3] + output_type
    save_videos(saved_name, video, duration=duration)










    
