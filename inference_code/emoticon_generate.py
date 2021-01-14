from utils import init_argument, warp_f
#from blended_image_generation import generate_image
from landmark_generate import landmark_detect, sol1_generate_landmark
from copy import copy

transform_dic = {'animation': 'params/baby-blended.pt', 'baby': 'params/disney-blended.pt', 'painting': 'params/metFaces-blended-32.pt'}

if __name__=="__main__":

    args = init_argument()
    # model_path = transform_dic[args.transform]
    # generate_image(args.file, model_path)

    if args.model == "sol1":
        first_landmark = landmark_detect(args.file)
        predicted_landmarks = sol1_generate_landmark(first_landmark.copy(), args.emotion)
        warp_f(args.file, args.type, first_landmark, predicted_landmarks, args.duration)

    elif args.model == "sol2":
        raise NotImplementedError
        sol2_generate_landmark(args.file, args.type, args.emotion, args.duration)

    else:
        assert "You have to selcet model in ['sol1', 'sol2']"



