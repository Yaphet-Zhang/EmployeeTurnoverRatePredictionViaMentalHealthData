# -*- coding: utf-8 -*-
# @Time    : 2021-10-21 14:20
# @Author  : zhangbowen


import pandas as pd
import numpy as np
from utils.hyperparameters import VAL_DATA, BATCH_SIZE
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader #, Dataset
from torchvision import transforms


########## read row data ##########
df = pd.read_csv(r'data/data_utf-8.csv', encoding='utf-8', low_memory=False) # for ignore DtypeWarning
# df.info()
print('csv table shape: {}'.format(df.shape))
print('========================================')


# list column names
column_names = df.columns.values.tolist()


########## pre-processing ##########
## features
feature_names = column_names[1:17]
print('feature names: ', feature_names)
data = np.array(df[feature_names]).astype(np.float32) # for cnn
data = data.reshape(-1, 1, 4, 4) # for cnn
print('========================================')

## label
label_name = column_names[-1]
print('label name: ', label_name)
label = np.array(df[label_name]).astype(np.float32) # classification(Cross-Entropy): is (n_samples, ) not (n_samples, 1)
print('========================================')


# train & validation
train_data, val_data, train_label, val_label = train_test_split(data, label, test_size = VAL_DATA)

# ndarray >> tensor
train_data = torch.tensor(train_data, dtype=torch.float32)
val_data = torch.tensor(val_data, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.int64) # classification(Cross-Entropy): int64
val_label = torch.tensor(val_label, dtype=torch.int64) # classification(Cross-Entropy): int64


########## get_data ##########
def get_data(BATCH_SIZE):
                
    # dataset
    train_dataset = TensorDataset(train_data, train_label)
    val_dataset = TensorDataset(val_data, val_label)
    
    # dataloader
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    val_loader = DataLoader(
        dataset = val_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle = False
    )

    return train_dataset, val_dataset, train_loader, val_loader


if __name__=='__main__':
    
    print('====================')
    print('data: {}'.format(data.shape))
    print('label: {}'.format(label.shape))
    print('====================')

    print('train data: {}'.format(train_data.shape))
    print('val data: {}'.format(val_data.shape))
    print('train label: {}'.format(train_label.shape))
    print('val label: {}'.format(val_label.shape))
    print('====================')

    ########## get_data ##########
    train_dataset, val_dataset, train_loader, val_loader = get_data(BATCH_SIZE)
    
    # dataset
    print('train dataset: {}'.format(train_dataset.__len__()))
    print('val dataset: {}'.format(val_dataset.__len__()))
    print('====================')

    # dataloader (batch size)
    for check_iteration, (check_data, check_label) in enumerate(train_loader):
        print('check batch')
        print('iteration: {}'.format(check_iteration)) 
        print('batch data: {}'.format(check_data.shape))
        print('batch label: {}'.format(check_label.shape))
        print('====================')
        break