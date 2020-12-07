#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:22:12 2020

@author: pratheek
"""


import scipy.io
import torch
import numpy as np
import random

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels.astype(np.float32)
        self.data = data.astype(np.float32)

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        X = np.asarray(self.data[index]).flatten()
        
        y = np.asarray(self.labels[index]).flatten()
       

        return X, y

def loadTrain(filename):  
    mat = scipy.io.loadmat("datasets/"+filename)
    
    if(filename == ('Soccer Player.mat')):
        target = mat["target"].T
        partials = mat["partial_target"].T
    else:
        target = mat["target"].todense().T
        partials = mat["partial_target"].todense().T
    
    data = mat["data"]
    tenth = int(data.shape[0]/10)
    
    combine = np.concatenate([data, partials, target], axis=1)
    
    
    random.shuffle(combine)
    data = combine[:, :data.shape[1]]
    partials = combine[:, data.shape[1]:data.shape[1]+partials.shape[1]]
    target = combine[:, data.shape[1]+partials.shape[1]:data.shape[1]+partials.shape[1]+target.shape[1]]
    
    train_dataset = Dataset(data[0:9*tenth], partials[0:9*tenth])
    test_dataset = Dataset(data[9*tenth:], target[9*tenth:])
    real_train_dataset = Dataset(data[0:9*tenth], target[0:9*tenth])
   
    return train_dataset, test_dataset, real_train_dataset, data.shape[1], partials.shape[1]

