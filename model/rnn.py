# -*- coding: utf-8 -*-
# @Time    : 2022-04-15 13:30
# @Author  : zhangbowen


import torch.nn as nn
import torch
import sys
sys.path.append('..')
from utils.hyperparameters import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True) # batch_first=True: x --> (batch_size, sequence_size, input_size)
        self.fc = nn.Linear(in_features=HIDDEN_SIZE, out_features=OUTPUT_SIZE)    

    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(device)

        x, _ = self.rnn(x, h0) # out: (batch_size, sequence_size, hidden_size)
        x = x[:, -1, :] # only use last time step for throughing the linear layer(classification), out: (batch_size, sequence_size, hidden_size) --> (batch_size, hidden_size)
        x = self.fc(x)

        return x


if __name__=='__main__':
    net = RNN().to(device)
    print(net)
    print('====================')
    x = torch.randn(100, 17, 8).to(device) # (batch_size, sequence_size, input_size)
    print('x: {}'.format(x.shape))
    y = net(x)
    print('y: {}'.format(y.shape))
    print('====================')
