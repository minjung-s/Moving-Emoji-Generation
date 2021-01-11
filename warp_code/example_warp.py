import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
import pandas as pd
from scipy.interpolate import Rbf
import cv2
import json

image_file_name = input() #ex)'004817.png'
img = cv2.imread(image_file_name)[:,:,::-1] # 이렇게 슬라이싱 해야 plt.imshow에서 파란색 이미지 안나옴

source_json = input() #ex)'004817.json'
f = open(source_json, 'r')
json_data = json.laod(f)
data = json_data[0]
f.close()

coords_from = np.array([[coord[0], coord[1]] for coord in data.values()], dtype=float)
#s는 배경을 유지하기 위해 꼭지점 등을 사용되는 좌표들
s = np.array([[0,0],[1024,1024], [1024,0], [0,1024],[0,512],[512,0],[1024,512],[512,1024]], dtype=float)
coords_from = np.vstack([coords_from, s])

target_json = input() #ex)'004817disgust.json'
f = open(target_json, 'r')
json_data = json.load(f)
f.close()

img_list = []
for i in range(50):
    target_data = json_data[str(i)] #i시점의 landmark
    coords_to = np.array([[coord[0], coord[1]] for coord in target_data.values()], dtype = float)
    coords_to = np.vstack([coords_to, s])
    warp_trans = PiecewiseAffineTransform()
    warp_trans.estimate(coords_to, coords_from)
    warped_face = warp(img, warp_trans)
    img_list.append(warped_face)
