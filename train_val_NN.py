# -*- coding: utf-8 -*-
# @Time    : 2021-10-21 20:50
# @Author  : zhangbowen


import torch
from dataset_NN import get_data
from utils.hyperparameters import BATCH_SIZE, EPOCH, NAME_NET
from model.nn import NN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time


########## device ##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'gpu':
    print('total gpu(s): {}'.format(torch.cuda.device_count()))
    print('gpu name(s): {}'.format(torch.cuda.get_device_name()))
    print('gpu spec(s): {} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))
print('device: {}'.format(device))
print('====================')


########## get data ##########
train_dataset, val_dataset, train_loader, val_loader = get_data(BATCH_SIZE)

# dataset
print('train dataset: {}'.format(train_dataset.__len__()))
print('val dataset: {}'.format(val_dataset.__len__()))

# dataloader (batch size)
for check_iteration, (check_data, check_label) in enumerate(train_loader):
    print('check batch')
    print('iteration: {}'.format(check_iteration)) 
    print('batch data: {}'.format(check_data.shape))
    print('batch label: {}'.format(check_label.shape))
    print('====================')
    break


########## network ##########
net = NN().to(device)
print(net)
print('====================')


########## loss function ##########
criterion = nn.CrossEntropyLoss()


########## optimization ##########
optimizer = optim.Adam(params=net.parameters())


########## training & validation ##########

# initialize loss & acc list (epochs)
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# initialize time (epochs)
train_time = 0
val_time = 0

# epoch
for i in range(EPOCH):
    print('--------------------')
    print('Epoch: {}/{}'.format(i+1, EPOCH))
    
    # initialize loss & acc (1 epoch)
    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0
    
    ########## training ##########
    # train start time (1 epoch)
    train_start_time = time.time()
    print('-----Train mode-----')
    net.train()
    
    # iteration
    for iteration, (inputs, labels) in enumerate(train_loader):
        # print('Iteration: {}'.format(iteration+1))
    
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()
        
        # sum loss (all iterations)
        train_loss += loss.item()

        # max logit >> predictions
        # predictions = torch.max(outputs, axis=1).indices
        predictions = torch.argmax(outputs, dim=1)

        # sum acc (all iterations)
        train_acc += torch.sum(predictions == labels).item() / len(labels)

    # mean loss (1 epoch)
    epoch_train_loss = train_loss / len(train_loader)
    print('Train loss: {:.4f}'.format(epoch_train_loss))

    # mean acc (1 epoch)
    epoch_train_acc = train_acc / len(train_loader)
    print('Train acc: {:.4f}'.format(epoch_train_acc))

    # loss list (epochs)
    train_loss_list.append(epoch_train_loss)

    # acc list (epochs)
    train_acc_list.append(epoch_train_acc)

    # train end time (1 epoch)
    train_end_time = time.time()
    print('time: {:.2f}s'.format(train_end_time - train_start_time))
    train_time += (train_end_time - train_start_time)

    ########## validation ##########
    # val start time (1 epoch)
    val_start_time = time.time()
    print('-----Val mode-----')
    net.eval()
    
    with torch.no_grad():
        
        # iteration
        for iteration, (inputs, labels) in enumerate(val_loader):
            # print('Iteration: {}'.format(iteration+1))
        
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # sum loss (all iterations)
            val_loss += loss.item()
        
            # max logit >> predictions
            # predictions = torch.max(outputs, axis=1).indices
            predictions = torch.argmax(outputs, dim=1)

            # sum acc (all iterations)
            val_acc += torch.sum(predictions == labels).item() / len(labels)

        # mean loss (1 epoch)  
        epoch_val_loss = val_loss / len(val_loader)
        print('Val loss: {:.4f}'.format(epoch_val_loss))
          
        # mean acc (1 epoch)
        epoch_val_acc = val_acc / len(val_loader)
        print('Val acc: {:.4f}'.format(epoch_val_acc))

        # loss list (epochs)
        val_loss_list.append(epoch_val_loss)

        # acc list (epochs)
        val_acc_list.append(epoch_val_acc)

        # val end time (1 epoch)
        val_end_time = time.time()
        print('time: {:.2f}s'.format(val_end_time - val_start_time))
        val_time += (val_end_time - val_start_time)


########## all time ##########
print('====================')
print('train all time: {:.2f}s'.format(train_time))
print('val all time: {:.2f}s'.format(val_time))
print('====================')


########## save weights ##########
weights_path = 'weights/' + NAME_NET + '_' + str(EPOCH) + '.pth'
torch.save(net.to(device).state_dict(), weights_path)
print(weights_path + ': saved!')
print('====================')


########## visualize loss ##########
plt.figure('loss')
plt.title('Train loss & Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# train loss
plt.plot(range(1, EPOCH+1), train_loss_list, 'b-', label='Train loss')
# val loss
plt.plot(range(1, EPOCH+1), val_loss_list, 'r-', label='Val loss')
plt.legend()
plt.savefig('weights/' + NAME_NET + '_' + str(EPOCH) + '_loss' + '.png')
plt.show()


########## visualize acc ##########
plt.figure('accuracy')
plt.title('Train acc & Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# train acc
plt.plot(range(1, EPOCH+1), train_acc_list, 'b-', label='Train acc')
# val acc
plt.plot(range(1, EPOCH+1), val_acc_list, 'r-', label='Val acc')
plt.legend()
plt.savefig('weights/' + NAME_NET + '_' + str(EPOCH) + '_acc' + '.png')
plt.show()
