#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:48:59 2021

@author: pratheek
"""


from dataset import Dataset, loadTrain, loadTrainT
from losses import cc_loss, min_loss, naive_loss, iexplr_loss, regularized_cc_loss, sample_loss_function, sample_reward_function, select_loss_function, select_reward_function
from networks import Prediction_Net, Prediction_Net_Linear, Selection_Net, Phi_Net

from train import q_accuracy_Subset
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import json
import pandas as pd
import os
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

epsilon = 1e-6

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(description = "Description for my parser")
parser.add_argument('--dump_dir', type=str, help="dump directory for results")
argument = parser.parse_args()

dump_dir = argument.dump_dir

log_file = "results/11022021/MSRCv2/3layer/cc_loss/9/logs/log.json"
dat = pd.read_json(log_file,orient = 'records',lines=True)
dat = dat.where(pd.notnull(dat), None)
data = dat.to_dict(orient='records')

#filename = 'lost'
#fold_no = 9
#
#model = '3layer'
#technique = "exponential_rl"
#pretrain_p = True
#pretrain_q = True
#pretrain_p_perc = None
k = 10
datasets = ["lost","Soccer Player", "MSRCv2", "BirdSong"]
models = ["1layer", "3layer"]
for filename in datasets:
    for model in models:
        
        directory = os.path.join(dump_dir, filename, model)
        print(directory)
        try:
            
            directories = os.listdir(directory)
            
        except:
            continue
        print(directories)
        directories = [x for x in directories if ('rl' in x)]
        
        for method in directories:
            for fold_no in range(10):
                print(method)
                train_dataset, real_train_dataset, val_dataset, real_val_dataset, test_dataset, real_test_dataset, input_dim, output_dim = loadTrain(filename+".mat", fold_no, k)
                            
                train_loader = torch.utils.data.DataLoader(train_dataset,
                  batch_size=batch_size_train, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset,
                  batch_size=batch_size_test, shuffle=False)
                val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=batch_size_test, shuffle=False)
                
                real_train_loader = torch.utils.data.DataLoader(real_train_dataset,
                  batch_size=batch_size_train, shuffle=False)
                real_test_loader = torch.utils.data.DataLoader(real_test_dataset,
                  batch_size=batch_size_test, shuffle=False)
                real_val_loader = torch.utils.data.DataLoader(real_val_dataset,
                  batch_size=batch_size_test, shuffle=False)
                
                if(model == "1layer"):
                    p_net = Prediction_Net_Linear(input_dim, output_dim)
                else:
                    p_net = Prediction_Net(input_dim, output_dim)  
                    
                p_net.to(device)
                p_optimizer = torch.optim.Adam(p_net.parameters())
                s_net = Selection_Net(input_dim, output_dim, True)
                s_net.to(device)
                
                overall_strategy = method
                if('linear' in overall_strategy):
                    technique = 'linear_rl'
                else:
                    technique = 'exponential_rl'
                """
                if(pretrain_p):
                    overall_strategy += "_P"
                    if(pretrain_p_perc is not None):
                        overall_strategy += pretrain_p_perc
                if(pretrain_q):
                    overall_strategy += "_Q"
                """
                
                dataset_technique_path = os.path.join(filename, model, overall_strategy, str(fold_no))
                
                result_filename = os.path.join(dump_dir, dataset_technique_path, "results", "out.txt")
                result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", "log.json")
                
                train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best.pth") 
                try:
                    checkpoint = torch.load(train_checkpoint)
                    print(overall_strategy)
                except:
                    #print("Absent: "+train_checkpoint)
                    continue
                p_net.load_state_dict(checkpoint['p_net_state_dict'])
                s_net.phi_net.load_state_dict(checkpoint['s_net_state_dict'])
                s_net.p_net.copy(p_net)
                q_surrogate_subset_train_acc = q_accuracy_Subset(train_loader, train_loader, s_net, technique)
                q_real_subset_train_acc = q_accuracy_Subset(real_train_loader, train_loader, s_net, technique)
                q_surrogate_subset_val_acc = q_accuracy_Subset(val_loader, val_loader, s_net, technique)
                q_real_subset_val_acc = q_accuracy_Subset(real_val_loader, val_loader, s_net, technique)
                q_surrogate_subset_test_acc = q_accuracy_Subset(test_loader, test_loader, s_net, technique)
                q_real_subset_test_acc = q_accuracy_Subset(real_test_loader, test_loader, s_net, technique)
        
                dat = pd.read_json(log_file,orient = 'records',lines=True)
                dat = dat.where(pd.notnull(dat), None)
                data = dat.to_dict(orient='records')
                
                for entry in data:
                    if(entry["epoch"] == -1):
                        entry['q_surrogate_subset_train_acc'] = q_surrogate_subset_train_acc
                        entry['q_real_subset_train_acc'] = q_real_subset_train_acc
                        entry['q_surrogate_subset_val_acc'] = q_surrogate_subset_val_acc
                        entry['q_real_subset_val_acc'] = q_real_subset_val_acc 
                        entry['q_surrogate_subset_test_acc'] = q_surrogate_subset_test_acc
                        entry['q_real_subset_test_acc'] = q_real_subset_test_acc
                    else:
                        entry['q_surrogate_subset_train_acc'] = None
                        entry['q_real_subset_train_acc'] = None
                        entry['q_surrogate_subset_val_acc'] = None
                        entry['q_real_subset_val_acc'] = None
                        entry['q_surrogate_subset_test_acc'] = None
                        entry['q_real_subset_test_acc'] = None
    
#dat.to_json()

#with open("test.json", "w") as jsonFile:
#    json.dump(data, jsonFile)
#with open("test.json", "w") as file:
#    for log in data:
#      json.dump(log, file)
#       file.write("\n")
#dat['fold'] = fold
#dat['model'] = model
#dat['model_arc'] = model_arc
#dat['dataset'] = dataset
#test_acc.append(dat[dat['epoch'] == -1])

#data["location"] = "NewPath"

#with open("replayScript.json", "w") as jsonFile:
#json.dump(data, jsonFile)
  
  
  
  
  
  
  
  
  
  
  
  