# https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

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

def landmark_detect(vid_path, vid_save_name):

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

    vid_in = cv2.VideoCapture(vid_path) #input video

    final=[]
    while True:
        try:
            # Get frame from video
            ret, image_o = vid_in.read()

           # resize the video
            image = cv2.resize(image_o, dsize=(1024, 1024), interpolation=cv2.INTER_AREA)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get faces (up-sampling=1)
            face_detector = detector(img_gray, 1)

            # make prediction and transform to numpy array
            landmarks = predictor(image, face_detector[0])  #68개 점 찾기
            #create list to contain landmarks
            landmark_list = []

            # append (x, y) in landmark_list
            for p in landmarks.parts():
                landmark_list.append([p.x, p.y])
                cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

            key_val = [ALL, landmark_list]
            landmark_dict = dict(zip(*key_val))
            final.append(landmark_dict)

            # wait for keyboard input
            key = cv2.waitKey(1)

            # if esc,
            if key == 27:
                break
                
        except Exception as e:
            break
    
    vid_in.release()
    cv2.destroyAllWindows()

    # save as json file
    data = {}
    for i in range(len(final)):
        data[i] = final[i]
    
    with open(vid_save_name, "w") as json_file:
        json_file.write(json.dumps(data))
        json_file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path" ,required= False, default = "dataset", help="video path")
    parser.add_argument("--save_path" ,required= False, default = "result", help="save video path")
    args = parser.parse_args()

    video_path = args.data_path
    num_list = os.listdir(video_path)

    c_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    video_list = []

    for num in num_list:
        class_list = os.listdir(os.path.join(video_path,num,num)) 

        for c in class_list:
            take_list = os.listdir(os.path.join(video_path,num,num,c)) 

            for t in take_list:
                file_list = os.listdir(os.path.join(video_path,num,num,c, t))
                for f in file_list:
                    if f.endswith(".avi"):
                        video_list.append(os.path.join(video_path, num, num, c, t, f))

    save_path = args.save_path

    for c in c_list:
        save_file_path = os.path.join(save_path,c)
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)

    for vid in video_list:
        for c in c_list:
            if c in vid:
                s_path = os.path.join(save_path,c) 
                file_name = c + str(len(os.listdir(s_path))+1) + '.json'
        all_save_path = os.path.join(s_path,file_name)
        landmark_detect(vid, all_save_path)

