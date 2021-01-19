from utils import init_argument, warp_f
from blended_image_generation import generate_image
from landmark_generate import landmark_detect, sol1_generate_landmark
from copy import copy

transform_dic = {'animation': 'params/animation.pt', 'baby': 'params/baby.pt', 'painting': 'params/painting.pt'}

if __name__=="__main__":

    args = init_argument()

    param_path = transform_dic[args.transform]

    # inversion and transform
    generate_image(args.file, param_path)
    
    # landmark of input image
    first_landmark = landmark_detect(args.file)

    # generate landmarks
    predicted_landmarks = sol1_generate_landmark(first_landmark.copy(), args.emotion)   

    # warping
    warp_f(args.file, args.type, first_landmark, predicted_landmarks, args.duration)
