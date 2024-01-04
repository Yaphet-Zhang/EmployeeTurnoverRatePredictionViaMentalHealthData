# -*- coding: utf-8 -*-
# @Time    : 2021-09-20 13:30
# @Author  : zhangbowen


import torch
from model.nn import NN
from dataset_NN import get_data
from utils.hyperparameters import EPOCH, BATCH_SIZE


########## device ##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'gpu':
    print('total gpu(s): {}'.format(torch.cuda.device_count()))
    print('gpu name(s): {}'.format(torch.cuda.get_device_name()))
    print('gpu spec(s): {} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))
print('device: {}'.format(device))
print('====================')


########## network ##########
net = NN().to(device)
print(net)
print('====================')


########## load weights ##########
weights_path = 'weights/classification' + str(EPOCH) + '.pth'
flag = net.load_state_dict(torch.load(weights_path, map_location=device))
print(flag)
print('====================')


########## get data ##########
train_dataset, val_dataset, train_loader, val_loader = get_data(BATCH_SIZE)
# .__getitem__ can be omitted
sample1_input = val_dataset.__getitem__(16)[0]
sample1_label = val_dataset.__getitem__(16)[1]
print('sample 1 data: {}'.format(sample1_input))
print('sample 1 label: {}'.format(sample1_label.item()))
print('====================')


########## inference ##########
net.eval()
with torch.no_grad():
    sample1_input = sample1_input.to(device)
    sample1_label = sample1_label.to(device)

    # forward
    sample1_output = net(sample1_input)

    # max logit >> prediction
    prediction = torch.argmax(sample1_output).item()
    print('sample 1 prediction: {}'.format(prediction))
    print('====================')
