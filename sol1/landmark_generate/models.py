import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data


class linear_Generator(nn.Module):
    def __init__(self, dim_z, n_out):
        super(linear_Generator, self).__init__()

        self.dim_z = dim_z
        self.n_out = n_out

        self.main = nn.Sequential(
            nn.Linear(dim_z, n_out*4),
            nn.BatchNorm1d(n_out*4),
            nn.ReLU(),

            nn.Linear(n_out*4, n_out*2),
            nn.BatchNorm1d(n_out*2),
            nn.ReLU(),

            nn.Linear(n_out*2, n_out),
        )

    def sample_z(self, num_samples):
        content = np.random.normal(0,1, (num_samples, self.dim_z))
        content = torch.from_numpy(content).float()
        if torch.cuda.is_available():
            content = content.cuda()

        return Variable(content)

    def sample_landmark(self, num_samples):
        z = self.sample_z(num_samples)
        h = self.main(z)

        return h

class linear_Discriminator(nn.Module):
    def __init__(self, n_input):
        super(linear_Discriminator, self).__init__()

        self.n_input = n_input

        self.main = nn.Sequential(
            nn.Linear(n_input, n_input*4),
            nn.BatchNorm1d(n_input*4),
            nn.ReLU(),

            nn.Linear(n_input*4, n_input*4),
            nn.BatchNorm1d(n_input*4),
            nn.ReLU(),

            nn.Linear(n_input*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x


class conv_Generator(nn.Module):
    def __init__(self, dim_z, n_out):
        super(conv_Generator, self).__init__()

        self.dim_z = dim_z
        self.n_out = n_out


        self.main = nn.Sequential(
            nn.ConvTranspose1d(dim_z, 64, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64,64, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64,32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, bias=False)
        )

    def sample_z(self, num_samples):
        content = np.random.normal(0,1, (num_samples, self.dim_z))
        content = torch.from_numpy(content).float()
        content = content.view(-1, self.dim_z, 1)
        if torch.cuda.is_available():
            content = content.cuda()

        return Variable(content)

    def sample_landmark(self, num_samples):
        z = self.sample_z(num_samples)
        h = self.main(z)

        return h

class conv_Discriminator(nn.Module):
    def __init__(self, n_input):
        super(conv_Discriminator, self).__init__()

        self.n_input = n_input

        self.main = nn.Sequential(
            nn.Conv1d(1,64, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64,64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 1, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2)
        )

        self.linear = nn.Sequential(
            nn.Linear(6,1),
            # nn.Sigmoid() #<- wgan때문에
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1,6)
        x = self.linear(x)
        return x
