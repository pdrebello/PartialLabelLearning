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
import pickle

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

class ConvDataset(torch.utils.data.Dataset):
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
        X = np.asarray(self.data[index])
        
        y = np.asarray(self.labels[index]).flatten()
       

        return X, y

   
def prepTrain(filename):  
    mat = scipy.io.loadmat("datasets/"+filename)
    if(filename == ('Soccer Player.mat')):
        target = mat["target"].T
        partials = mat["partial_target"].T
    else:
        target = mat["target"].todense().T
        partials = mat["partial_target"].todense().T
    
    data = mat["data"]
    
    
    combine = np.concatenate([data, partials, target], axis=1)
    
    np.random.shuffle(combine)
    dat = combine[:, :data.shape[1]]
    partials = combine[:, data.shape[1]:data.shape[1]+partials.shape[1]]
    target = combine[:, data.shape[1]+partials.shape[1]:data.shape[1]+partials.shape[1]+target.shape[1]]
    
    
    
    
    with open("datasets/"+filename+".pkl", "wb") as f:
        pickle.dump(dat, f)
        pickle.dump(partials, f)
        pickle.dump(target, f)
    #tr = list(dat)
    #for count,i in enumerate(tr):
     #   for j in range(count+1, len(tr)):
    #        if((tr[count] == tr[j]).all()):
    #            print(j)
    #train_dataset = Dataset(data[0:9*tenth], partials[0:9*tenth])
    #test_dataset = Dataset(data[9*tenth:], target[9*tenth:])
    #real_train_dataset = Dataset(data[0:9*tenth], target[0:9*tenth])

    #return train_dataset, test_dataset, real_train_dataset, data.shape[1], partials.shape[1]



def loadTrainT(filename):  
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
    #return data[0:9*tenth], data[9*tenth:]
    return train_dataset, test_dataset, real_train_dataset, data.shape[1], partials.shape[1]

def loadTrain(filename, fold_no, k):  
    
    
    
    with open("datasets/"+filename+".pkl", "rb") as f:
        data = pickle.load(f)
        partials = pickle.load(f)
        target = pickle.load(f)
    
    split = int(data.shape[0]/k)
    
    train_data_list = []
    train_target_list = []
    train_partials_list = []
    
    test_data_list = []
    test_target_list = []
    test_partials_list = []
    
    val_data_list = []
    val_target_list = []
    val_partials_list = []
    
    for i in range(k):
        if(fold_no == i):
            test_data_list.append(data[i*split : (i+1)*split])
            test_target_list.append(target[i*split : (i+1)*split])
            test_partials_list.append(partials[i*split : (i+1)*split])
        elif(i == (fold_no+1)%k):
            val_data_list.append(data[i*split : (i+1)*split])
            val_target_list.append(target[i*split : (i+1)*split])
            val_partials_list.append(partials[i*split : (i+1)*split])
        else:
            train_data_list.append(data[i*split : (i+1)*split])
            train_target_list.append(target[i*split : (i+1)*split])
            train_partials_list.append(partials[i*split : (i+1)*split])
    
    train_data = np.vstack(train_data_list)
    train_target = np.vstack(train_target_list)
    train_partials = np.vstack(train_partials_list)
    
    test_data = np.vstack(test_data_list)
    test_target = np.vstack(test_target_list)
    test_partials = np.vstack(test_partials_list)
    
    val_data = np.vstack(val_data_list)
    val_target = np.vstack(val_target_list)
    val_partials = np.vstack(val_partials_list)
    
    train_dataset = Dataset(train_data, train_partials)
    test_dataset = Dataset(test_data, test_target)
    val_dataset = Dataset(val_data, val_target)
    real_train_dataset = Dataset(train_data, train_target)
    #return train_data, test_data
    return train_dataset, test_dataset, real_train_dataset, val_dataset, data.shape[1], partials.shape[1]


def loadTrainAnalysis(filename, fold_no, k):  
    
    
    
    with open("datasets/"+filename+".pkl", "rb") as f:
        data = pickle.load(f)
        partials = pickle.load(f)
        target = pickle.load(f)
    
    split = int(data.shape[0]/k)
    
    train_data_list = []
    train_target_list = []
    train_partials_list = []
    
    test_data_list = []
    test_target_list = []
    test_partials_list = []
    
    val_data_list = []
    val_target_list = []
    val_partials_list = []
    
    for i in range(k):
        if(fold_no == i):
            test_data_list.append(data[i*split : (i+1)*split])
            test_target_list.append(target[i*split : (i+1)*split])
            test_partials_list.append(partials[i*split : (i+1)*split])
        elif(i == (fold_no+1)%k):
            val_data_list.append(data[i*split : (i+1)*split])
            val_target_list.append(target[i*split : (i+1)*split])
            val_partials_list.append(partials[i*split : (i+1)*split])
        else:
            train_data_list.append(data[i*split : (i+1)*split])
            train_target_list.append(target[i*split : (i+1)*split])
            train_partials_list.append(partials[i*split : (i+1)*split])
    
    dic = {}
    
    dic["train_data"] = torch.from_numpy(np.vstack(train_data_list)).float()
    dic["train_target"] = torch.from_numpy(np.vstack(train_target_list)).float()
    dic["train_partials"] = torch.from_numpy(np.vstack(train_partials_list)).float()
    
    dic["test_data"] = torch.from_numpy(np.vstack(test_data_list)).float()
    dic["test_target"] = torch.from_numpy(np.vstack(test_target_list)).float()
    dic["test_partials"] = torch.from_numpy(np.vstack(test_partials_list)).float()
    
    dic["val_data"] = torch.from_numpy(np.vstack(val_data_list)).float()
    dic["val_target"] = torch.from_numpy(np.vstack(val_target_list)).float()
    dic["val_partials"] = torch.from_numpy(np.vstack(val_partials_list)).float()
    
    
    #train_dataset = Dataset(train_data, train_partials)
    #test_dataset = Dataset(test_data, test_partials)
    #val_dataset = Dataset(val_data, val_partials)
    #real_train_dataset = Dataset(train_data, train_target)
    #real_test_dataset = Dataset(test_data, test_target)
    #real_val_dataset = Dataset(val_data, val_target)
    #return train_data, test_data
    return dic, data.shape[1], partials.shape[1]

#a = loadTrain('MSRCv2.mat',4,10)
#tr = list(a[0])
#te = list(a[1])

#for count,i in enumerate(tr):
#    for j in range(count+1, len(tr)):
#        if((tr[count] == tr[j]).all()):
#            print(j)
    #print(i.shape)
    #print(te[0].shape)
#for i in tr:
    #print(type(i))
    #print(type(te[0]))
#    if((te[0] == i).all()):
#        print(te[0])
#        print(i)

#datasets = ['MSRCv2','Yahoo! News','lost','Soccer Player','BirdSong']
#for i in datasets:
#    prepTrain(i+".mat")
#a = loadTrain("lost.mat")