# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:48:11 2024

@author: sbhowmi2
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def visualize(x):
    '''
        x: (n_channels, height, width) expected to be torch tensors
    '''
    fig = plt.imshow(np.array(x.detach()).transpose(1,2,0))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
def standardize_data(X):
    #X_standardized = (X-np.min(X))/(np.max(X) - np.min(X))
    X_standardized = X.reshape(X.shape[0], X.shape[-2], X.shape[-1])
    X_standardized = (X_standardized-np.mean(X_standardized))/np.std(X_standardized)
    return X_standardized.reshape(X.shape[0], 1, X.shape[-2], X.shape[-1])