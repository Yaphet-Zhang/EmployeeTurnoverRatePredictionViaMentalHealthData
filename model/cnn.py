# -*- coding: utf-8 -*-
# @Time    : 2021-10-21 18:30
# @Author  : zhangbowen


import torch.nn as nn
import torch
from torchvision import models
import sys
sys.path.append('..')
from utils.hyperparameters import OUTPUT_SIZE


# vgg: 50 * 50 = 2500 features over (needed)
# vgg = models.vgg16(pretrained=True)
# vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# vgg.classifier[6] = nn.Linear(in_features=4096, out_features=OUTPUT_SIZE)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(),
            # need to manually calculate C*H*W(convolution) >> in_features(Linear)
            nn.Linear(in_features=16*4*4, out_features=10),
            # nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(10, OUTPUT_SIZE),
        )


    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten: reshape(N, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x



if __name__=='__main__':
    
    net = CNN()
    print(net)
    print('====================')

    x = torch.randn(32, 1, 4, 4)
    print('x: {}'.format(x.shape))
    y = net(x)
    print('y: {}'.format(y.shape))
    print('====================')
