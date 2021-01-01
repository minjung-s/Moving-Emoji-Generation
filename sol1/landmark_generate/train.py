import pandas as pd
import numpy as np
from PIL import Image
import os

import torch.nn.utils.spectral_norm as spectral_norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import models

df = pd.read_csv("../landmark_example.csv", index_col=0)

datas = np.array(df)
datas = datas/500

data_loader = DataLoader(datas, batch_size=32, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

netG = models.conv_Generator(16, 70).to(device)
netD = models.conv_Discriminator(70).to(device)


# optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
# optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
optimizerD = optim.RMSprop(netD.parameters(), lr = 0.00005)
optimizerG = optim.RMSprop(netG.parameters(), lr = 0.00005)

num_epochs = 10
clip_value = 0.01

for epoch in range(num_epochs):
    for i, data in enumerate(data_loader):

        #train discriminator
        netD.zero_grad()
        real = data.to(device)
        real = real.float()
        real= real.view(-1,1,70)
        b_size = real.size(0)
        d_real = netD(real).view(-1)


        fake = netG.sample_landmark(b_size)
        d_fake = netD(fake.detach()).view(-1)
        errD = -torch.mean(d_real) + torch.mean(d_fake)
        errD.backward()
        optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-clip_value, clip_value)

        #train Generator
        if i % 5 == 0:
            netG.zero_grad()
            fake = netG.sample_landmark(b_size)
            d_fake = netD(fake).view(-1)
            errG = -torch.mean(d_fake)
            errG.backward()
            D_G_z2 = d_fake.mean().item()
            optimizerG.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(data_loader),
                     errD.item(), errG.item()))

    netG.eval()
    torch.save(netG, os.path.join('../logs', 'generator{}.pytorch'.format(epoch)))
