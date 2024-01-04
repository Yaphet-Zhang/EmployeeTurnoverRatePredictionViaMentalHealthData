import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


########## normalization ##########
def normalization(data):
    scale = np.max(abs(data))
    return data/scale


########## standardization ##########
standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
standardization = standard_scaler.fit_transform


########## visualization ##########
def confusion_matrix(y_true, y_pred):
    tp = np.sum((np.array(y_true)==1) & (np.array(y_pred)==1))
    tn = np.sum((np.array(y_true)==0) & (np.array(y_pred)==0))
    fp = np.sum((np.array(y_true)==0) & (np.array(y_pred)==1))
    fn = np.sum((np.array(y_true)==1) & (np.array(y_pred)==0))
    confusion_matrix = np.array([[tp, fp],
                                [fn, tn]])
    # heatmap
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    indices = range(len(confusion_matrix))
    plt.xticks(indices, ['1:turnover', '0:working'])
    plt.yticks(indices, ['1:turnover', '0:working'])
    plt.colorbar()
    plt.title('confusion matrix')
    plt.xlabel('True')
    plt.ylabel('Predict')
    # show
    for first_index in range(len(confusion_matrix)): # row
        for second_index in range(len(confusion_matrix[first_index])): # column
            plt.text(first_index, second_index, confusion_matrix[second_index][first_index])
    plt.show()