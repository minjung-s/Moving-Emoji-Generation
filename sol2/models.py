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
            SpatioTemporalConv(ndf * 2,ndf * 4, 4, stride=[1, 2, 2], padding=[0, 1, 1], bias=False),
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


class CategoricalVideoDiscriminator(VideoDiscriminator): #Video Discriminator상속받아 사용
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

class embedding_f(nn.Module): 
    #U-Net에 쓰일 임베딩 class. 1D vector -> 2D Embedding vector
    def __init__(self,n_class, x):
      super().__init__()
      self.n_class = n_class
      self.categ_embed = nn.Embedding(self.n_class, x)

    def forward(self,z_category):
      categ_embedding = self.categ_embed(z_category.long())
      return categ_embedding


class UNet(nn.Module): 
    """
    z_motion,z_category(one-hot)은 입력으로 받는다.
    이미지는 인코더를 통과하여 임베딩
    이미지 임베딩벡터와 z_motion,z_category와 concate되어 디코더 통과
    디코더 피쳐맵에, 인코더 피쳐맵과 z_motion임베딩값 z_cateogry임베딩값 concate
    """
    def __init__(self, n_class, n_channels, z_motion, batch_size,video_length):
        super().__init__()

        self.n_class = n_class
        self.z_noise = torch.from_numpy(np.random.normal(0, 1, (batch_size, 100)).astype(np.float32))
        self.z_motion = z_motion
        #self.z_category = z_category_labels
        self.n_channels = n_channels
        self.video_length = video_length

        self.embedding_c1 = embedding_f(self.n_class,16)
        #self.embedding_m1 = embedding_f(int(torch.max(self.z_motion).item()),16)

        self.embedding_c2 = embedding_f(self.n_class,64)
        #self.embedding_m2 = embedding_f(int(torch.max(self.z_motion).item()),64)

        self.embedding_c3 = embedding_f(self.n_class,256)
        #self.embedding_m3 = embedding_f(int(torch.max(self.z_motion).item()),256)

        self.embedding_c4 = embedding_f(self.n_class,1024)
        #self.embedding_m4 = embedding_f(int(torch.max(self.z_motion).item()),1024)
        
        # input 3x64x64 
        self.conv_down1 = nn.utils.spectral_norm(nn.Conv2d(3, 16, 4, stride =2,padding=1)) # 32x32x16 if input1024 ->Output = 512x512x16/conv층 더 쌓기
        self.conv_down_acf1 = nn.LeakyReLU(inplace=True)

        self.conv_down2 = nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride =2,padding=1)) # 16x16x32 
        self.conv_down_acf2 = nn.LeakyReLU(inplace=True)

        self.conv_down3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride =2,padding=1)) # 8x8x64
        self.conv_down_acf3 = nn.LeakyReLU(inplace=True)

        self.conv_down4 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride =2,padding=1)) #4x4x128
        self.conv_down_acf4 = nn.LeakyReLU(inplace=True)      
         
        self.flatten = nn.Flatten() 
        self.linear = nn.Linear(2048,200) #image embedding dim = 200
        #여기까지 이미지 임베딩을 위한 인코더, 밑에부터 디코더

        self.conv_up4 = nn.utils.spectral_norm(nn.ConvTranspose2d(316, 128, 4, 1, padding=0)) #output feature map W,H = 4
        self.conv_up_acf4 = nn.LeakyReLU(inplace=True)

        self.conv_up3 = nn.utils.spectral_norm(nn.ConvTranspose2d(259, 64, 4, 2, padding=1)) #output feature map W,H = 8
        self.conv_up_acf3 = nn.LeakyReLU(inplace=True)

        self.conv_up2 = nn.utils.spectral_norm(nn.ConvTranspose2d(131, 32, 4, 2, padding=1)) #output feature map W,H = 16
        self.conv_up_acf2 = nn.LeakyReLU(inplace=True)
        
        self.conv_up1 = nn.utils.spectral_norm(nn.ConvTranspose2d(67, 16, 4, 2, padding=1))#output feature map W,H = 32
        self.conv_up_acf1 = nn.LeakyReLU(inplace=True)

        self.conv_last = nn.ConvTranspose2d(35, self.n_channels,  4, 2, padding=1) #output feature map W,H = 64
        
        
    def forward(self,image,z_category):
        conv1 = self.conv_down1(image)
        conv1 = self.conv_down_acf1(conv1)

        conv2 = self.conv_down2(conv1)
        conv2 = self.conv_down_acf2(conv2)
        
        conv3 = self.conv_down3(conv2)
        conv3 = self.conv_down_acf3(conv3)
        
        conv4 = self.conv_down4(conv3)
        conv4 = self.conv_down_acf4(conv4)

        x = self.flatten(conv4)
        x = self.linear(x)

        if torch.cuda.is_available():
            z_category = z_category.cuda()
            self.z_motion = self.z_motion.cuda()
            x = x.cuda()
            self.z_noise = self.z_noise.cuda()

        x = x.repeat(self.video_length,1)
        z_noise_1 = self.z_noise.repeat(self.video_length,1)

        p = torch.cat([z_category, self.z_motion, x, z_noise_1], dim=1)
        # x : 200, noise : 10, z_motion: 13, categ_embedding : 3
        p = p.view(p.size(0),p.size(1),1,1)#[b,316,1,,]

        u_conv4 = self.conv_up4(p)
        x = self.conv_up_acf4(u_conv4)#c=128
        conv4 = conv4.repeat(self.video_length,1,1,1)
        categ_embedding_1 = self.embedding_c1(z_category)
        categ_embedding_1 = categ_embedding_1.reshape(categ_embedding_1.size(0),categ_embedding_1.size(1),x.size(2),x.size(3))
        x = torch.cat([x, conv4, categ_embedding_1], dim=1) 
        
        x = self.conv_up_acf3(u_conv3)
        conv3 = conv3.repeat(self.video_length,1,1,1)
        categ_embedding_2 = self.embedding_c2(z_category)
        categ_embedding_2 = categ_embedding_2.reshape(categ_embedding_2.size(0),categ_embedding_2.size(1),x.size(2),x.size(3))
        x = torch.cat([x, conv3, categ_embedding_2 ], dim=1)
        
        u_conv2 = self.conv_up2(x)
        x = self.conv_up_acf2(u_conv2)
        conv2 = conv2.repeat(self.video_length,1,1,1)
        categ_embedding_3 = self.embedding_c3(z_category)
        categ_embedding_3 = categ_embedding_3.reshape(categ_embedding_3.size(0),categ_embedding_3.size(1),x.size(2),x.size(3))
        x = torch.cat([x, conv2, categ_embedding_3 ], dim=1)

        u_conv1 = self.conv_up1(x)
        x = self.conv_up_acf1(u_conv1)
        conv1 = conv1.repeat(self.video_length,1,1,1)
        
        categ_embedding_4 = self.embedding_c4(z_category)
        categ_embedding_4 = categ_embedding_4.reshape(categ_embedding_4.size(0),categ_embedding_4.size(1),x.size(2),x.size(3))
        x = torch.cat([x, conv1, categ_embedding_4], dim=1)

        out = self.conv_last(x)

        return out


class VideoGenerator(nn.Module):
    def __init__(self, n_class,n_channels, dim_z_category, dim_z_motion,
                 video_length, ngf=64):
        super(VideoGenerator, self).__init__()
        self.n_class = n_class

        self.n_channels = n_channels
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        dim_z = dim_z_motion + dim_z_category

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

    def sample_z_m(self, num_samples, video_len=None): #GRU통과한 motion vector 만들기
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples, video_len): # category one-hot vector, z_category_labels(categorical classification loss에 사용) 만들기
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

        return Variable(one_hot_video), torch.from_numpy(classes_to_generate)


    def sample_z_video(self, num_samples, video_len=None): 
        # motion(z)만들기, motion(z:생성에 사용)와 one hot category(z_category:생성에 사용) z_category_labels(categorical classification loss에 사용) 출력 
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
        z_motion = self.sample_z_m(num_samples, video_len)

        if z_category is not None:
            z = torch.cat([z_category, z_motion], dim=1)
        else:
            z = z_motion
        return z, z_category, z_category_labels

    def sample_videos(self, image, num_samples, target_class=None, video_len=None): # main network(Unet)으로 video 만들기
        video_len = video_len if video_len is not None else self.video_length
        self.video_length
        z,  z_category, z_category_labels = self.sample_z_video(num_samples, video_len)
        if target_class is not None : #inference
            print("inference")
            z_category = target_class


        main = UNet(self.n_class, self.n_channels, z, num_samples,video_len)
        if torch.cuda.is_available():
          main.cuda()
        h = main(image,z_category)
        h = h.view(int(h.size(0) / video_len), int(video_len), self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4)
        return h, Variable(z_category_labels, requires_grad=False)
    
    def sample_images(self, image, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        z,  z_category, z_category_labels = self.sample_z_video(num_samples, video_len)

        h, _ = self.sample_videos(image, num_samples, None, video_len)
        h_result =[]
        for i in range(num_samples):
          j = np.random.randint(video_len)
          img_frame = h[i,:,j,:,:]
          h_result.append(img_frame)
        h_result = torch.stack(h_result)

        return h_result, None


    def get_gru_initial_state(self, num_samples): #z_motion만드는 recurrent(GRU cell) network input
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples): #z_motion만드는 recurrent(GRU cell) network input
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())
