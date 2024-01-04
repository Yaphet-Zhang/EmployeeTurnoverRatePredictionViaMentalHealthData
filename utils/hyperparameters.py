# -*- coding: utf-8 -*-
# @Time    : 2021-10-21 16:20
# @Author  : zhangbowen


import numpy as np

########## common ##########
BATCH_SIZE = 64
EPOCH = 1000
VAL_DATA = 0.2

## Feature Selection
# START_FEATURE, END_FEATURE = 1, 20 # 19 features
# START_FEATURE, END_FEATURE = 20, -1 # 17 features
START_FEATURE, END_FEATURE = 1, -1 # 137 features or all
# START_FEATURE, END_FEATURE = 1, -2 # 136 features (for RNN/LSTM/GRU = 8*17)

SELECTED_FEATURES = ['B-222-1022(勤務日の平均的な就寝時刻を教えて下さい)',
                     'B-80-317(自分の仕事に誇りを感じる)',
                     'B-37-145(体のふしぶしが痛む)',
                     'B-185-861(仕事におけるルールや判断基準が会社として整備されている)',
                     'B-9-33(自分で仕事の順番・やり方を決めることができる)',
                     'B-199-916(会社の経営指針（理念・ビジョン・行動指針など）に満足だ)',
                     'B-221-1019(仕事に忙殺され、自分の自由に使える時間が十分に取れない)',
                     'B-209-957(入社前の期待と入社後の現実でどの程度ギャップがありますか？)']


########## nn ##########
# INPUT_SIZE = 137 # number of features
# OUTPUT_SIZE = 2 # number of label
# NAME_NET = 'nn' 


########## cnn ##########
# INPUT_SIZE = 137
# OUTPUT_SIZE = 2
# NAME_NET = 'cnn'


########## rnn ##########
# # sum number of features(136) = INPUT_SIZE(8) * SEQUENCE_SIZE(17)
# INPUT_SIZE = 8
# SEQUENCE_SIZE = 17
# HIDDEN_SIZE = 32 # customize
# NUM_LAYERS = 2 # number of RNN for stacking
# OUTPUT_SIZE = 2
# NAME_NET = 'rnn'


########## lstm ##########
# sum number of features(136) = INPUT_SIZE(8) * SEQUENCE_SIZE(17)
INPUT_SIZE = 8
SEQUENCE_SIZE = 17
HIDDEN_SIZE = 32 # customize
NUM_LAYERS = 2 # number of RNN for stacking
OUTPUT_SIZE = 2
NAME_NET = 'lstm'


########## gru ##########
# # sum number of features(136) = INPUT_SIZE(8) * SEQUENCE_SIZE(17)
# INPUT_SIZE = 8
# SEQUENCE_SIZE = 17
# HIDDEN_SIZE = 32 # customize
# NUM_LAYERS = 2 # number of RNN for stacking
# OUTPUT_SIZE = 2
# NAME_NET = 'gru'


########## xgboost ##########
PARAMS = {
    'booster': 'gbtree', # model: GBDT
    'objective': 'binary:logistic', # 2分类
    # 'objective': 'multi:softmax', # 多分类
    # 'num_class': 2, # 类数, 与'multi:softmax'并用
    # 'objective': 'multi:softprob', # 多分类概率
    'eval_metric': 'auc',
    # 'eval_metric': 'logloss',
    'max_depth': 5, # 6 构建树的深度, 越大越容易过拟合 / 12
    'min_child_weight': 3, # 1 节点的最少特征数, 越小越容易过拟合  
    'gamma': 0.4, # 0 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 /0.05
    'subsample': 1, # 1 采样训练数据，设置为0.5，随机选择一般的数据实例
    'colsample_bytree': 1, # 1 构建树时的采样比率
    'lambda': 1, # 1 L2 正则项系数, 越大越不容易过拟合 /450
    'alpha': 0, # 1 L1 正则项系数, 越大越不容易过拟合
    'eta': 0.3, # 0.3 learning rate
    # 'scale_pos_weight': (346/1300),  # 1 用来处理正负样本不均衡的问题,通常取：sum(negative cases) / sum(positive cases)
    # 'seed': 0,
    # 'random_state':1000,
    # 'missing': 1,
    'silent': 0, # 设置成1则没有运行信息输出，最好是设置为0
    # 'nthread': 4, # cpu 线程数
}
NUM_ROUND = 5000 # 迭代次数
EARLY_ROUND = 5000 # 迭代次数(没有明显变化即刻停止)


########## knn ##########
K_RANGE = np.arange(1,20)
K = 7


########## PCA ##########
N_COMPONENTS = 120 # numbers of PC
CUTOFF = 346
# for visualization
FIRST_PC, SECOND_PC = 0, 1 # must < N_COMPONENTS


########## Sparse PCA ##########
# N_COMPONENTS = 120 # numbers of PC
# ALPHA = 0.0001
# CUTOFF = 346
# # for visualization
# FIRST_PC, SECOND_PC = 0, 1 # must < N_COMPONENTS


########## Kernel PCA ##########
# N_COMPONENTS = 120 # numbers of PC
# KERNEL = 'rbf'
# GAMMA = 1/137 # 1/feature number
# CUTOFF = 346
# # for visualization
# FIRST_PC, SECOND_PC = 0, 1 # must < N_COMPONENTS


########## K-means ##########
NUM_CLUSTERS = 5
INIT = 'k-means++'
N_INIT = 20
MAX_ITER = 300
TOL = 0.0001