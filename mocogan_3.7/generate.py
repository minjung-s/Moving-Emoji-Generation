import os
import torch
import matplotlib.pyplot as plt
import numpy as np

generator = torch.load('generator_100000.pytorch', map_location={'cuda:0':'cpu'})
generator.eval()
# v, _ = generator.sample_videos(1, 16)
# v.shape
# v = v.detach()
# video = [v[:,:,i,:,:] for i in range(16)]
# s = video[2]
# s = s[0]
# s *= 255
# s = s.type(torch.uint8)
# s.shape
# plt.imshow(s.permute(1,2,0))

def videos_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

v, _ = generator.sample_videos(1,16)
video = videos_to_numpy(v).squeeze().transpose((1,2,3,0))

for i in range(16):
    temp = video[i]
    plt.imshow(temp)
    plt.show()
