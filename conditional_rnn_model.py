import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class conditional_rnn(nn.Module):

    def __init__(self, input_size= 136, hidden_size=136, t_stamp = 36, output_size = 136, num_condition = 3, conditions = None):
        super(conditional_rnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.t_stamp = t_stamp
        self.output_size = output_size
        self.num_condition = num_condition
        self.conditions = conditions

        if self.conditions == 'concat':
            self.embedding = nn.Embedding(num_condition, num_condition)
            self.fc1 = nn.Linear(input_size+num_condition, t_stamp*input_size)
        else:
            self.embedding = nn.Embedding(num_condition, input_size)
            self.fc1 = nn.Linear(input_size, t_stamp*input_size)

        self.rnn = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.Linear(t_stamp*hidden_size, t_stamp*output_size)

    def forward(self, x, cond):
        #condition concat
        embed = self.embedding(cond.long())
        if self.conditions == 'concat':
            x = torch.cat((x, embed), dim = 1)
        elif self.conditions == 'element_sum':
            x = x+embed
        else:  #element_mul
            x = x*embed

        #forward fc1 -> rnn -> fc2
        outputs = F.relu(self.fc1(x))
        outputs = outputs.view(x.size(0), self.t_stamp, self.input_size)
        outputs, _ = self.rnn(outputs)
        outputs = outputs.view(x.size(0), -1)
        outputs = self.fc2(outputs)

        return outputs