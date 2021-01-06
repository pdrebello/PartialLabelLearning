#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:41:01 2020

@author: pratheek
"""

import pickle
import numpy as np
import random

#return true with probability percent
def flip_coin(percent=0.1):
    percent = percent*100
    return random.randrange(100) < percent

def make_target(query, target, number, method, count):
    
    if(method == "shuffle"):
        indices = np.argwhere(query == 0).flatten()
        old_indices = indices.copy()
        
        results = [target]
        
        for j in range(number):
            random.shuffle(indices)
            new_target = target.copy()
            for i in range(len(old_indices)):
                new_target[indices[i]] = target[old_indices[i]]
            results.append(new_target)
        random.shuffle(results)
        return results
    
    elif(method == "switch"):
        indices = np.argwhere(query == 0).flatten()
        results = [target]
        for j in range(number):
            new_target = target.copy()
            for k in range(count):
                samp = random.sample(list(indices), 2)
                temp = new_target[samp[0]]
                new_target[samp[0]] = new_target[samp[1]]
                new_target[samp[1]] = temp
            results.append(new_target)
        random.shuffle(results)
        return results
    
    elif(method == "flip"):
        indices = np.argwhere(query == 0).flatten()
        results = [target]
        
        options = [1,2,3,4,5,6,7,8,9]
        for j in range(number):
            new_target = target.copy()
            for k in list(indices):
                if(flip_coin(count)):
                    new_target[k] = random.choice(options)
                    
            results.append(new_target)
        random.shuffle(results)
        return results
    
def make_noisy_target(query, target, number, method, count):
    indices = np.argwhere(query == 0).flatten()
    
    results = []
    options = [1,2,3,4,5,6,7,8,9]
    for j in range(number):
        new_target = target.copy()
        for k in list(indices):
            if(flip_coin(count)):
                new_target[k] = random.choice(options)
                
        results.append(new_target)
    random.shuffle(results)
    return results        
        


filenames = ["sudoku_9_train_e_unq_10k.pkl"]

random.seed(10)

for fn in filenames:
    
    for prob in [0.2,0.4,0.6,0.8]:
        new_data = []
        with open(fn,'rb') as f:
            data = pickle.load(f)
            
            for sample in data:
                new_sample = sample.copy()
                
                #Commands for partial_labels
                #new_target_set = make_target(sample["query"], sample["target_set"][0],5)
                #new_target_set = make_target(sample["query"], sample["target_set"][0],5, "shuffle", 8)
                
                #Command for noisy target
                new_target_set = make_noisy_target(sample["query"], sample["target_set"][0],1, "flip", prob)
                
                new_sample["target_set"] = new_target_set
                new_sample["count"] = len(new_target_set)
                new_data.append(new_sample)
                
                
        with open("sudoku_9_train_e_unq_noisy_10k_shuffle_"+str(prob)+".pkl",'wb') as f:
            pickle.dump(new_data,f)
        
        
