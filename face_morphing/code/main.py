from face_landmark_detection import generate_face_correspondences
from delaunay_triangulation import make_delaunay
from face_morph import generate_morph_sequence

import subprocess
import argparse
import shutil
import os
import imageio
import cv2
import numpy as np

import json


def doMorphing(img1, landmark2, num, save_image_path):

	# points2 = landmark2
	[size, img1, points1, points2, list3] = generate_face_correspondences(img1, landmark2)
	tri = make_delaunay(size[1], size[0], list3)

	generate_morph_sequence(img1, points1, points2, tri, size, save_image_path, num)

def img_file_to_gif(img_list, output_file_name):
    ## imge 파일 리스트로부터 gif 생성 
    imgs_array = [np.array(imageio.imread(img_file)) for img_file in img_list]

    imageio.mimsave(output_file_name, imgs_array, duration=0.5)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--img1" ,required= True, help="The First Image")
	parser.add_argument("--landmark_path" ,required= True, help="landmark dir")
	parser.add_argument("--save_image_path" ,required= True, help="save_image dir")
	parser.add_argument("--output", default='results/emoticon.gif',help="Output Video Path")
	
	# parser.add_argument("--img1" ,default = 'images/aligned_images/001228_01.png', help="The First Image")
	# parser.add_argument("--landmark_path", default = 'landmark', help="landmark dir")
	# parser.add_argument("--save_image_path",default = 'save_image', help="save_image dir")
	# parser.add_argument("--output", default='results/emoticon.gif',help="Output Video Path")
	
	args = parser.parse_args()

	image1 = cv2.imread(args.img1)
	
	landmark_dir = args.landmark_path 
	file_list = os.listdir(landmark_dir)

	if not os.path.exists(args.save_image_path):
		os.makedirs(args.save_image_path)

	# input landmark 형식에 따라 바꿔야 함
	for i in range(len(file_list)):
		# json file 불러오기 
		file_path = os.path.join(landmark_dir,file_list[i])

		with open(file_path, 'r') as f:
			json_data = json.load(f)

		landmark = json_data

		for se in range(len(landmark)):
			landmark_list = []
			for l in range(len(landmark[str(se)])):
				landmark_list.append(landmark[str(se)][str(l)])
			# print(landmark_list)
			doMorphing(image1, landmark_list, se, args.save_image_path)


		img_list = os.listdir(args.save_image_path)
		img_path_list = []
		for img in img_list:
			img_path_list.append(os.path.join(args.save_image_path,img))
		img_file_to_gif(img_path_list, args.output)

		print('complete')

		#### sample image delete 
		for img_file in img_path_list:
			if os.path.exists(img_file):
				os.remove(img_file)
		print('image file delete complete')
