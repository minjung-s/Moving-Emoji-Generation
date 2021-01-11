"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Generates multiple videos given a model and saves them as video files using ffmpeg

Usage:
    generate_videos.py [options] <model> <input_img> <target class> <output_folder>

Options:
    -n, --num_videos=<count>                number of videos to generate [default: 1]
    -o, --output_format=<ext>               save videos as [default: gif]
    -f, --number_of_frames=<count>          generate videos with that many frames [default: 50]

    --ffmpeg=<str>                          ffmpeg executable (on windows should be ffmpeg.exe). Make sure
                                            the executable is in your PATH [default: ffmpeg]
"""

import os
import docopt
import torch
from PIL import Image

from trainers import videos_to_numpy

import subprocess as sp


def save_video(ffmpeg, video, filename):
    command = [ffmpeg,
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', '64x64',
               '-pix_fmt', 'rgb24',
               '-r', '8',
               '-i', '-',
               '-c:v', 'mjpeg',
               '-q:v', '3',
               '-an',
               filename]

    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    pipe.stdin.write(video.tostring())


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    # generate_videos.py <model> <input_img> <target class> <output_folder>
    # ex) generate_videos.py ./generator.pt ./input_img.png disgust ./ouput
    # target clss = disgust / happiness / surprise
    generator = torch.load(args["<model>"], map_location={'cuda:0': 'cpu'})
    generator.eval()
    num_videos = int(args['--num_videos'])
    output_folder = args['<output_folder>']
    img_path = args['<input_image>']
    target_class = args['<target_class>'] #0,1,2로 입력



    img = Image.open(img_path)
    img_tmp = np.array(img)
    image = torch.from_numpy(img_tmp)#input img -> tensor

    if target_class == "disgust" :
        target_class_onehot = torch.from_numpy(np.array[1,0,0])
    elif target_class == "happiness" :
        target_class_onehot = torch.from_numpy(np.array[0,1,0]) 
    else :
        target_class_onehot = torch.from_numpy(np.array[0,0,1])


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_videos):
        v, _ = generator.sample_videos(image,1, target_class_onehot,int(args['--number_of_frames']))
        video = videos_to_numpy(v).squeeze().transpose((1, 2, 3, 0))
        save_video(args["--ffmpeg"], video, os.path.join(output_folder, "{}.{}".format(i, args['--output_format'])))
