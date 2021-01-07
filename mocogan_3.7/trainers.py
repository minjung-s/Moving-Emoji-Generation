import os
import time

import numpy as np


import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.optim as optim

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def videos_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def one_hot_to_class(tensor):
    a, b = np.nonzero(tensor)
    return np.unique(b).astype(np.int32)

class Trainer(object):
    def __init__(self, image_sampler, video_sampler, log_interval, train_batches, log_folder, use_cuda=False,
                 use_infogan=True, use_categories=True):

        self.use_categories = use_categories

        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.category_criterion = nn.CrossEntropyLoss()

        self.image_sampler = image_sampler
        self.video_sampler = video_sampler

        self.video_batch_size = self.video_sampler.batch_size
        self.image_batch_size = self.image_sampler.batch_size

        self.log_interval = log_interval
        self.train_batches = train_batches

        self.log_folder = log_folder

        self.use_cuda = use_cuda
        self.use_infogan = use_infogan

        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        self.image_enumerator = None
        self.video_enumerator = None

    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    def sample_real_image_batch(self):
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.image_sampler)

        batch_idx, batch = next(self.image_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.items():
                b[k] = v.cuda()

        if batch_idx == len(self.image_sampler) - 1:
            self.image_enumerator = enumerate(self.image_sampler)

        return b

    def sample_real_video_batch(self):
        if self.video_enumerator is None:
            self.video_enumerator = enumerate(self.video_sampler)

        batch_idx, batch = next(self.video_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.items():
                b[k] = v.cuda()

        if batch_idx == len(self.video_sampler) - 1:
            self.video_enumerator = enumerate(self.video_sampler)

        return b

    #학습을 위해(이미지끼리 비교를 해야하므로) 좌표에서 랜드마크 이미지로 변환
    def coord_to_arr(self,fake_batch):
        landmark_images = []
        for batch in fake_batch:
            img = np.zeros((64,64))
            for coord in batch.reshape(68,2):
                x = int(coord[0]*32 + 32)
                y = int(coord[1]*32 + 32)
                img[y,x] = 1
            landmark_images.append(img)
        length = len(fake_batch)
        return torch.tensor(landmark_images, dtype = torch.float32).view(length,1,64,64)

    def coord_to_arr_video(self,fake_batch):
        landmark_videos = []
        for batch in fake_batch:
            landmark_images = []
            for t in batch:
                img = np.zeros((64,64))
                for coord in t.reshape(68,2):
                    x = int(coord[0]*32 + 32)
                    y = int(coord[1]*32 + 32)
                    img[y,x] = 1
                landmark_images.append(img)
            landmark_videos.append(landmark_images)
        length = len(fake_batch)
        video_length_temp = len(batch)
        return torch.tensor(landmark_videos, dtype = torch.float32).view(length, 1,video_length_temp, 64,64)

    def train_discriminator(self, discriminator, sample_true, sample_fake, opt, batch_size, use_categories):
        opt.zero_grad()

        real_batch = sample_true()
        batch = Variable(real_batch['images'], requires_grad=False)

        # util.show_batch(batch.data)

        fake_batch, generated_categories = sample_fake(batch_size)

        if fake_batch.dim() == 2:
            fake_batch = self.coord_to_arr(fake_batch)
        else:
            fake_batch = self.coord_to_arr_video(fake_batch)

        # fake_batch.cuda()
        fake_batch = fake_batch.to(self.device)

        real_labels, real_categorical = discriminator(batch)
        fake_labels, fake_categorical = discriminator(fake_batch.detach())

        ones = self.ones_like(real_labels)
        zeros = self.zeros_like(fake_labels)

        l_discriminator = self.gan_criterion(real_labels, ones) + \
                          self.gan_criterion(fake_labels, zeros)

        if use_categories:
            # Ask the video discriminator to learn categories from training videos
            categories_gt = Variable(torch.squeeze(real_batch['categories'].long()), requires_grad=False)
            l_discriminator += self.category_criterion(real_categorical.squeeze(), categories_gt)

        l_discriminator.backward()
        opt.step()

        return l_discriminator

    def train_generator(self,
                        image_discriminator, video_discriminator,
                        sample_fake_images, sample_fake_videos,
                        opt):

        opt.zero_grad()

        # train on images

        fake_batch, generated_categories = sample_fake_images(self.image_batch_size)
        fake_batch = self.coord_to_arr(fake_batch)
        # fake_batch.cuda()
        fake_batch = fake_batch.to(self.device)
        fake_labels, fake_categorical = image_discriminator(fake_batch)
        all_ones = self.ones_like(fake_labels)

        l_generator = self.gan_criterion(fake_labels, all_ones)

        # train on videos

        fake_batch, generated_categories = sample_fake_videos(self.video_batch_size)
        fake_batch = self.coord_to_arr_video(fake_batch)
        # fake_batch.cuda()
        fake_batch = fake_batch.to(self.device)
        fake_labels, fake_categorical = video_discriminator(fake_batch)
        all_ones = self.ones_like(fake_labels)

        l_generator += self.gan_criterion(fake_labels, all_ones)

        if self.use_infogan:
            # Ask the generator to generate categories recognizable by the discriminator
            l_generator += self.category_criterion(fake_categorical.squeeze(), generated_categories)

        l_generator.backward()
        opt.step()

        return l_generator

    def train(self, generator, image_discriminator, video_discriminator):
        if self.use_cuda:
            generator.cuda()
            image_discriminator.cuda()
            video_discriminator.cuda()


        # create optimizers
        opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        opt_image_discriminator = optim.Adam(image_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)
        opt_video_discriminator = optim.Adam(video_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)

        # training loop

        def sample_fake_image_batch(batch_size):
            return generator.sample_images(batch_size)

        def sample_fake_video_batch(batch_size):
            return generator.sample_videos(batch_size)

        def init_logs():
            return {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}

        batch_num = 0

        logs = init_logs()

        start_time = time.time()

        while True:
            generator.train()
            image_discriminator.train()
            video_discriminator.train()

            opt_generator.zero_grad()

            opt_video_discriminator.zero_grad()

            # train image discriminator
            l_image_dis = self.train_discriminator(image_discriminator, self.sample_real_image_batch,
                                                   sample_fake_image_batch, opt_image_discriminator,
                                                   self.image_batch_size, use_categories=False)

            # train video discriminator
            l_video_dis = self.train_discriminator(video_discriminator, self.sample_real_video_batch,
                                                   sample_fake_video_batch, opt_video_discriminator,
                                                   self.video_batch_size, use_categories=self.use_categories)

            # train generator
            l_gen = self.train_generator(image_discriminator, video_discriminator,
                                         sample_fake_image_batch, sample_fake_video_batch,
                                         opt_generator)

            logs['l_gen'] += l_gen.data.item()

            logs['l_image_dis'] += l_image_dis.data.item()
            logs['l_video_dis'] += l_video_dis.data.item()

            batch_num += 1

            if batch_num % self.log_interval == 0:

                log_string = "Batch %d" % batch_num
                for k, v in logs.items():
                    log_string += " [%s] %5.3f" % (k, v / self.log_interval)

                log_string += ". Took %5.2f" % (time.time() - start_time)

                print(log_string)


                logs = init_logs()
                start_time = time.time()

                generator.eval()

                torch.save(generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))

            if batch_num >= self.train_batches:
                torch.save(generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))
                break
