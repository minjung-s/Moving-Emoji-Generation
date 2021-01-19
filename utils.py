import argparse
import os
import imageio
import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

def init_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file", 
        required=True, 
        type=str,
        help="Image name you want to transform"
        ) 

    parser.add_argument(
        "--transform", 
        required=True, 
        type=str,
        choices=['animation', 'baby', 'painting'],
        help="Choose transform type"
        ) 

    parser.add_argument(
        "--emotion", 
        default='happiness', 
        type=str,
        choices=['disgusted', 'happiness', 'surprised'],
        help="Choose your emotion (default: happiness)"
        ) # disgust, happiness, surprise
    
    parser.add_argument(
        "--type", 
        default='gif', 
        type=str,
        choices=['gif', 'mp4'],
        help="Format of output video (default: gif)"
        ) 

    parser.add_argument(
        "--duration", 
        default=0.1, 
        type=float,
        help="Video duration"
        ) 

    args = parser.parse_args()

    return args


def warp_f(file_path, output_type, first_landmark, predicted_landmarks, duration=0.1):
    first_landmark = first_landmark[0]
    saved_name = file_path[:-3] + output_type

    img = cv2.imread(file_path)[:,:,::-1]
    
    coords_from = np.array([[coord[0], coord[1]] for coord in first_landmark.values()], dtype=float)

    s = np.array([[0,0],[1024,1024], [1024,0], [0,1024],[0,512],[512,0],[1024,512],[512,1024]], dtype=float)
    coords_from = np.vstack([coords_from, s])

    warped_images = []
    for i in range(len(predicted_landmarks)):
        target_data = predicted_landmarks[str(i)] #i시점의 landmark
        coords_to = np.array([[coord[0], coord[1]] for coord in target_data.values()], dtype = float)
        coords_to = np.vstack([coords_to, s])
        warp_trans = PiecewiseAffineTransform()
        warp_trans.estimate(coords_to, coords_from)
        warped_face = warp(img, warp_trans)

        warped_images.append(warped_face)

    warped_images = np.array(warped_images)
    save_videos(saved_name, warped_images, duration=duration) 


def videos_to_numpy(video):
    generated = video.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8').squeeze().transpose((1, 2, 3, 0))


def save_videos(saved_name, images, duration):
    imageio.mimsave(saved_name, images, duration=duration) 
