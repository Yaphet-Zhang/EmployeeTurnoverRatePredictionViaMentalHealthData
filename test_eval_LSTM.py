# -*- coding: utf-8 -*-
# @Time    : 2022-04-22 11:50
# @Author  : zhangbowen


import torch
from model.lstm import LSTM
from dataset_LSTM import get_data
from utils.hyperparameters import EPOCH, BATCH_SIZE, NAME_NET, INPUT_SIZE, SEQUENCE_SIZE
import torch.nn as nn
from utils.utils import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


########## device ##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print('total gpu(s): {}'.format(torch.cuda.device_count()))
    print('gpu name(s): {}'.format(torch.cuda.get_device_name()))
    print('gpu spec(s): {} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))
print('device: {}'.format(device))
print('====================')


########## network ##########
net = LSTM().to(device)
print(net)
print('====================')


########## load weights ##########
weights_path = 'weights/' + NAME_NET + '_' + str(EPOCH) + '.pth'
flag = net.load_state_dict(torch.load(weights_path, map_location=device))
print(flag)
print('====================')


########## get data ##########
train_dataset, val_dataset, train_loader, val_loader = get_data(BATCH_SIZE)
print('test set: {}'.format(len(val_dataset)))
print('====================')


########## loss function ##########
criterion = nn.CrossEntropyLoss()


########## test ##########
net.eval()

with torch.no_grad():
    
    # initialize loss & acc (all)
    test_loss = 0
    test_acc = 0
    
    # initialize labels & predictions for precision/ recall/ f1-score/ AUC/ confusion matrix/ ROC curve... (all)
    test_labels = []
    test_preds = []

    for input, label in val_dataset:

        input = input.to(device)
        input = input.reshape(-1, SEQUENCE_SIZE, INPUT_SIZE) # for RNN input (batch_size, sequence_size, input_size)
        label = label.to(device)

        # forward
        output = net(input)
        output = output.view(1,-1)
        label = label.view(1)
        loss = criterion(output, label)
        
        # sum loss (all)
        test_loss += loss.item()

        # max logit >> predictions
        prediction = torch.argmax(output, dim=1)

        # sum acc (all)
        test_acc += torch.sum(prediction == label)

        # summary labels & predictions into list
        test_labels.append(label.item())
        test_preds.append(prediction.item())

    # mean loss    
    mean_loss = test_loss / len(val_dataset)
    print('test loss: {:.4f}'.format(mean_loss))

    # mean acc
    mean_acc = test_acc / len(val_dataset)
    print('test acc: {:.4f}'.format(mean_acc))
print('====================')


########## evaluation ##########
y_true = test_labels
y_pred = test_preds

# acc
print('val acc:')
val_acc = accuracy_score(y_true, y_pred)
print(val_acc)

# precision
print('val precision:')
val_precision = precision_score(y_true, y_pred)
print(val_precision)

# recall
print('val recall:')
val_recall = recall_score(y_true, y_pred)
print(val_recall)

# f1 score
print('val f1 score:')
val_f1_score = f1_score(y_true, y_pred)
print(val_f1_score)

# AUC
print('val AUC:')
val_auc = roc_auc_score(y_true, y_pred)
print(val_auc)

# confusion matrix
confusion_matrix(y_true, y_pred)

