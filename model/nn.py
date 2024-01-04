# -*- coding: utf-8 -*-
# @Time    : 2021-10-21 18:30
# @Author  : zhangbowen


import torch.nn as nn
import torch
import sys
sys.path.append('..')
from utils.hyperparameters import INPUT_SIZE, OUTPUT_SIZE


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=INPUT_SIZE, out_features=10),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm2d() # regularization
            nn.ReLU(),
        ) 
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=10),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=10, out_features=OUTPUT_SIZE),
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        
        return output 

if __name__=='__main__':
    
    net = NN()
    print(net)
    print('====================')

    x = torch.randn(64, 137)
    print('x: {}'.format(x.shape))
    y = net(x)
    print('y: {}'.format(y.shape))
    print('====================')