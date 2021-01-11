
import subprocess
import argparse
import shutil
import os
import imageio
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data, io
import pandas as pd
from scipy.interpolate import Rbf
import json
from PIL import Image


def doWarping(img1, source_landmark, landmarks, save_image_path):

    # points2 = landmark2
    coords_from = np.array([[coord[0], coord[1]] for coord in source_landmark.values()], dtype=float)
    #s는 배경을 유지하기 위해 꼭지점 등을 사용되는 좌표들
    s = np.array([[0,0],[1024,1024], [1024,0], [0,1024],[0,512],[512,0],[1024,512],[512,1024]], dtype=float)
    coords_from = np.vstack([coords_from, s])

    for i in range(50):
        target_data = landmarks[str(i)] #i시점의 landmark
        coords_to = np.array([[coord[0], coord[1]] for coord in target_data.values()], dtype = float)
        coords_to = np.vstack([coords_to, s])
        warp_trans = PiecewiseAffineTransform()
        warp_trans.estimate(coords_to, coords_from)
        warped_face = warp(img1, warp_trans)

        io.imsave(save_image_path + '/'+str(i) + '.png', warped_face)

	

def img_file_to_gif(img_list, output_file_name):
    ## imge 파일 리스트로부터 gif 생성 
    imgs_array = [np.array(imageio.imread(img_file)) for img_file in img_list]

    imageio.mimsave(output_file_name, imgs_array, duration=0.2) # 수정


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--img1" ,required= True, help="The First Image")
    # parser.add_argument("--landmark_path" ,required= True, help="landmark dir")
    # parser.add_argument("--save_image_path" ,required= True, help="save_image dir")
    # parser.add_argument("--output", default='results/emoticon.gif',help="Output Video Path")

    parser.add_argument("--img1" ,default = 'data/001228.png', help="The First Image") #required로 바꾸기
    parser.add_argument("--landmark_path", default = 'landmark', help="landmark dir") #required로 바꾸기
    parser.add_argument("--save_image_path",default = 'save_image', help="save_image dir") #required로 바꾸기
    parser.add_argument("--output", default='results',help="Output Video Path") #required로 바꾸기
    parser.add_argument("--output_file", default = 'gif' ,help="Output Video file (mp4/gif)")

    args = parser.parse_args()

    # image1 = cv2.imread(args.img1)
    img = cv2.imread(args.img1)[:,:,::-1]

    landmark_dir = args.landmark_path 
    file_list = os.listdir(landmark_dir)

    # source_landmark = args.s_landmark
    source_landmark = "source_landmark/001228.json"

    if not os.path.exists(args.save_image_path):
        os.makedirs(args.save_image_path)

    # input landmark 형식에 따라 바꿔야 함
    for i in range(len(file_list)):
        # json file 불러오기 
        file_path = os.path.join(landmark_dir,file_list[i])

        source_json = source_landmark
        f = open(source_json, 'r')
        json_data = json.load(f)
        json_data = json_data[0]
        f.close()

        f = open(file_path, 'r')
        landmark_list = json.load(f)
        f.close()

        doWarping(img, json_data, landmark_list, args.save_image_path)

        img_list = os.listdir(args.save_image_path)
        img_path_list = []
        for img in img_list:
            img_path_list.append(os.path.join(args.save_image_path,img))

        if args.output_file == 'gif':
            img_file_to_gif(img_path_list, args.output + '/' + file_list[i] + '.gif')

        print('complete')

        #### sample image delete 
        for img_file in img_path_list:
            if os.path.exists(img_file):
                os.remove(img_file)
        print('image file delete complete')