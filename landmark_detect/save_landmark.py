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
            image = cv2.resize(image_o, dsize=(480, 480), interpolation=cv2.INTER_AREA)
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
    parser.add_argument("--video_path" ,required= False, default = "vid_test", help="video path")
    parser.add_argument("--save_path" ,required= False, default = "result", help="save video path")
    args = parser.parse_args()

    video_path = args.video_path
    dataset_list = os.listdir(video_path)


    for data_name in dataset_list:

        vid_list = os.listdir(os.path.join(video_path,data_name))

        save_path = os.path.join(args.save_path,data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if len(vid_list)==0:
            assert False, "no video"

        for vid_name in vid_list:
            vid_path =  os.path.join(video_path,data_name,vid_name)
            temp = vid_name.split('.') # 저장할 image name
            vid_save_name = temp[0] + '.json'
            all_save_path = os.path.join(save_path, vid_save_name)
            
            landmark_detect(vid_path, all_save_path)

