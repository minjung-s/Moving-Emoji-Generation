import torch
import torch.nn as nn
import numpy as np


class one2many_lstm(nn.Module):

    def __init__(self, input_size = 136, hidden_size = 136, t_stamp = 20, output_size = 136):
        super(one2many_lstm,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.t_stamp = t_stamp
        self.output_size = output_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

        self.rnn = nn.LSTM(input_size, hidden_size)

    def expanding(self, x):
        extra = torch.zeros((x.size(0), self.t_stamp-1,self.input_size))
        extra = extra.to(self.device)
        x = x.view(x.size(0),1,-1)
        expanded = torch.cat((x, extra), dim = 1)
        return expanded

    def forward(self, x):
        inputs = self.expanding(x)
        recurrent, _ = self.rnn(inputs)
        recurrent = recurrent.view(x.size(0),-1)
        return recurrent


# one to many rnn 어떻게 하는지 물어보기

'''
class one2many_lstm(nn.Module):

    def __init__(self, input_size = 136, hidden_size = 136, t_stamp = 20, output_size = 136):
        super(one2many_lstm,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.t_stamp = t_stamp
        self.output_size = output_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

        self.rnn = nn.LSTM(input_size, hidden_size)

    # def expanding(self, x):
    #     extra = torch.zeros((x.size(0), self.t_stamp-1,self.input_size))
    #     extra = extra.to(self.device)
    #     x = x.view(x.size(0),1,-1)
    #     expanded = torch.cat((x, extra), dim = 1)
    #     return expanded

    def forward(self, x):
        inputs = x.view(x.size(0),1,-1)
        inputs = inputs.repeat_interleave(self.t_stamp, dim = 1)
        recurrent, _ = self.rnn(inputs)
        recurrent = recurrent.view(x.size(0),-1)
        return recurrent
'''
