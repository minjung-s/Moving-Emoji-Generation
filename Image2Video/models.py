"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import math
from torch.nn.modules.utils import _triple

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(ImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchVideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(PatchVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None



class SpatioTemporalConv(nn.Module):
    
    #12.20 Relu->LeakyRelU
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.leakyrelu = nn.LeakyReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.leakyrelu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma
        #self.SpatioTemporalConv = SpatioTemporalConv()???

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            SpatioTemporalConv(n_channels, ndf, 4, stride=[1, 2, 2], padding=[0, 1, 1], bias=False),
            #nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            SpatioTemporalConv(ndf, ndf * 2, 4, stride=[1, 2, 2], padding=[0, 1, 1], bias=False),
            #nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            SpatioTemporalConv(ndf * 2, 4, stride=[1, 2, 2], padding=[0, 1, 1], bias=False),
            #nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            SpatioTemporalConv(ndf * 4, ndf * 8, 4, stride=[1, 2, 2], padding=[0, 1, 1], bias=False),
            #nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            SpatioTemporalConv(ndf * 8, n_output_neurons, 4, stride=1, padding=0, bias=False),
            #nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None


class CategoricalVideoDiscriminator(VideoDiscriminator):
    def __init__(self, n_channels, dim_categorical, n_output_neurons=1, use_noise=False, noise_sigma=None):
        super(CategoricalVideoDiscriminator, self).__init__(n_channels=n_channels,
                                                            n_output_neurons=n_output_neurons + dim_categorical,
                                                            use_noise=use_noise,
                                                            noise_sigma=noise_sigma)

        self.dim_categorical = dim_categorical

    def split(self, input):
        return input[:, :input.size(1) - self.dim_categorical], input[:, input.size(1) - self.dim_categorical:]

    def forward(self, input):
        h, _ = super(CategoricalVideoDiscriminator, self).forward(input)
        labels, categ = self.split(h)
        return labels, categ

# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.LeakyReLU(inplace=True),
#     ) 

def embedding_f(n_class, x, z_category):
    categ_embed = Embedding(n_class, np.prod(x.shape))
    categ_embedding = categ_embed(z_category)
    categ_embedding = torch.flatten(categ_embedding, start_dim = x.shape[0], end_dim = x.shape[2])
    return categ_embedding


class UNet(nn.Module):

    def __init__(self, n_class, n_channels, z_motion, image, z_category_labels, batch_size):
        super().__init__()
        
        self.z_noise = np.random.normal(0, 1, (n_class, self.dim_z_content)).astype(np.float32) #차원 설정해줘야 함
        self.z_motion = z_motion
        self.z_category = z_category_labels
        #Variable(LongTensor(np.random.randint(0, n_class, batch_size)))
        self.n_channels = n_channels
        self.image = image

        
        
        # input 512x512x3으로 가정
        self.conv_down1 = nn.utils.spectral_norm(nn.Conv2d(3, 16, 4, stride =2,padding=0)) # Output = 255x255x16
        self.conv_down_acf1 = nn.LeakyReLU(inplace=True)

        self.conv_down2 = nn.utils.spectral_norm(nn.Conv2d(16, 32, 3,stride =1, padding=0)) # Output = 252x252x32
        self.conv_down_acf2 = nn.LeakyReLU(inplace=True)

        self.conv_down3 = nn.utils.spectral_norm(nn.Conv2d(64, 64, 4, stride =2,padding=0)) #124x124x64
        self.conv_down_acf3 = nn.LeakyReLU(inplace=True)

        self.conv_down4 = nn.utils.spectral_norm(nn.Conv2d(128, 128, 3,stride =1, padding=0)) #121x121x128
        self.conv_down_acf4 = nn.LeakyReLU(inplace=True)      
         
        self.flatten = nn.Flatten() 
        self.linear = nn.Linear(200) #image embedding dim = 200

        self.conv_up4 = nn.utils.spectral_norm(nn.ConvTranspose2d(256 + 512, 256, 3, padding=1)) # 여기 input, output size 바꿔주기!
        self.conv_up_acf4 = nn.LeakyReLU(inplace=True)

        self.conv_up3 = nn.utils.spectral_norm(nn.ConvTranspose2d(128 + 256, 128, 3, padding=1))
        self.conv_up_acf3 = nn.LeakyReLU(inplace=True)

        self.dconv_up2 = nn.utils.spectral_norm(nn.ConvTranspose2d(128 + 64, 64, 3, padding=1))
        self.conv_up_acf2 = nn.LeakyReLU(inplace=True)
        
        self.conv_up1 = nn.utils.spectral_norm(nn.ConvTranspose2d(64 + 32, 32, 3, padding=1))
        self.conv_up_acf1 = nn.LeakyReLU(inplace=True)

        self.conv_last = nn.ConvTranspose2d(16, self.n_channels, 1)
        
    def forward(self):
        x = self.image

        conv1 = self.conv_down1(x)
        conv1 = self.conv_down_acf1(conv1)

        conv2 = self.conv_down2(conv1)
        conv2 = self.conv_down_acf2(conv2)
        
        conv3 = self.dconv_down3(conv2)
        conv3 = self.conv_down_acf3(conv3)
        
        conv4 = self.dconv_down4(conv3)
        conv4 = self.conv_down_acf4(conv4)
        
        x = self.linear(self.flatten(conv4))

        categ_embedding = embedding_f(n_class, np.prod(x.shape), self.z_category)
        
        p = torch.cat([torch.reshape(categ_embedding,(6,)), self.z_motion, x, torch.reshape(self.z_noise,(100,))], dim=0)
        # x : 200, noise : 100, z_motion: ???, categ_embedding : 6

        u_conv4 = self.conv_up4(p)
        x = self.conv_up_acf4(x)

        categ_embedding = embedding_f(n_class, np.prod(x.shape), self.z_category)        
        x = torch.cat([x, conv4, torch.reshape(categ_embedding,x.shape), torch.reshape(self.z_motion,x.shape)], dim=1) 

        u_conv3 = self.conv_up3(x)
        x = self.conv_up_acf3(x)

        categ_embedding = embedding_f(n_class, np.prod(x.shape), self.z_category)      
        x = torch.cat([x, conv3, torch.reshape(categ_embedding,x.shape), torch.reshape(self.z_motion,x.shape)], dim=1)    

        u_conv2 = self.conv_up2(x)
        x = self.conv_up_acf2(x)

        categ_embedding = embedding_f(n_class, np.prod(x.shape), self.z_category)       
        x = torch.cat([x, conv2, torch.reshape(categ_embedding,x.shape), torch.reshape(self.z_motion,x.shape)], dim=1)  

        u_conv1 = self.conv_up1(x)
        x = self.conv_up_acf1(x)

        categ_embedding = embedding_f(n_class, np.prod(x.shape), self.z_category)       
        x = torch.cat([x, conv1, torch.reshape(categ_embedding,x.shape), torch.reshape(self.z_motion,x.shape)], dim=1) 

        out = self.conv_last(x)
        
        return out

class VideoGenerator(nn.Module):
    def __init__(self, n_class, n_channels, image, batch_size, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=64): # input 수정 필요? (batch size 불러오기 필요할 듯?)
        super(VideoGenerator, self).__init__()

        self.n_class = n_class
        self.n_channels = n_channels
        self.image = image
        self.batch_size = batch_size
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length
        

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)
        self.main = UNet(n_class, n_channels, z_motion, image, batch_size) # z_motion ?
        
        """
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        """

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples, video_len):
        video_len = video_len if video_len is not None else self.video_length

        if self.dim_z_category <= 0:
            return None, np.zeros(num_samples)

        classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    """
    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)
    """

    def sample_z_video(self, image, z_content, num_samples, video_len=None):
        #z_content = !!!!input image!!!!
        #z_content = self.sample_z_content(num_samples, video_len)
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
        z_motion = self.sample_z_m(num_samples, video_len)

        if z_category is not None:
            z = torch.cat([z_category, z_motion], dim=1)
        else:
            z = z_motion

        return z, z_category_labels

    def sample_videos(self, image, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        image = self.image

        z, z_category_labels = self.sample_z_video(num_samples, video_len) #z_motion, z_cat

        h = self.main(self.n_class, self.n_channels, z_motion, image, z_category_labels, self.batch_size) #UNet(n_class, n_channels, z_motion, image, batch_size): # z_motion ?

        h = h.view(h.size(0) / video_len, video_len, self.n_channels, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category_labels)

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()

        h = h.permute(0, 2, 1, 3, 4)
        return h, Variable(z_category_labels, requires_grad=False)

    def sample_images(self, num_samples):
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h, None

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())
