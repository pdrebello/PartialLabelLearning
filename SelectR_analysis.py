#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:07:20 2021

@author: pratheek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:57:04 2020

@author: pratheek
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#import torch_optimizer as optim
import scipy.io
from dataset import Dataset, loadTrain, loadTrainT, loadTrainAnalysis
import sys
from IPython.core.debugger import Pdb
import random
import csv
import os

n_epochs = 150
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.001
momentum = 0.5
log_interval = 10

epsilon = 1e-6

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cuda:1")
#print(device)
#print(torch.cuda.current_device())

vals = [[],[],[],[]]

def cc_loss(output, target):
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.log(loss+epsilon)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def naive_reward(output, target):
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    #loss = torch.log(loss)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def select_loss_function(p, q):
    loss = (q*torch.log(p+epsilon)).sum(dim=1).mean()
    return -loss

def select_reward_function(p, q):
    rew = (q*p).sum(dim=1).mean()
    return -rew

def sample_reward_function(p, q, a, mask):
    
    rew = (a*p).sum(dim=1) - ((1-a)*mask*p).sum(dim=1)
    #lap = sum (log (ai*qi + (1-ai)*(1-qi)))
    lap = a*torch.log(q+ epsilon) + (1-a)*mask*torch.log(1-q+ epsilon)
    lap = lap.sum(dim=1)
    #(lap * rew)
    #lap = torch.log(lap+epsilon).sum(dim=1)
    
    return -torch.mean(rew * lap)

def sample_loss_function(p, a):
    loss = torch.log((a*p).sum(dim=1) +epsilon)
    return -torch.mean(loss)

class Prediction_Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Prediction_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.softmax(self.fc3(x))
        return x
    
    def copy(self, net2):
        self.load_state_dict(net2.state_dict())
    

class Phi_Net(nn.Module):
    def __init__(self, input_dim, output_dim, input_x):
        super(Phi_Net, self).__init__()
        if(input_x):
            self.fc1 = nn.Linear(input_dim+output_dim, 512)
        else:
            self.fc1 = nn.Linear(input_dim, 512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x, p, targetSet, rl_technique):
        mask = targetSet>0
        
        p = p * targetSet
        
        if(input_x):
            x = torch.cat((x, p), 1)
        
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x[~mask] = float('-inf')
        if(rl_technique == "sample"):
            x = F.sigmoid(x)
        else:
            x = F.softmax(x)
        return x
    
    def copy(self, net2):
        self.load_state_dict(net2.state_dict())

class Selection_Net(nn.Module):
    def __init__(self, input_dim, output_dim, input_x):
        super(Selection_Net, self).__init__()
        
        self.p_net = Prediction_Net(input_dim, output_dim)
        for param in self.p_net.parameters():
            param.requires_grad = False
            
        self.phi_net = Phi_Net(input_dim, output_dim, input_x)

    def forward(self, x, targetSet, rl_technique):
        with torch.no_grad():
            p = self.p_net(x)
        
        x = self.phi_net(x, p, targetSet, rl_technique)
        return x


def test(test_data, input_x):
    p_net.eval()
    
    pred_list = []
    s_pred_list = []
    targ_pred_list = []
    with torch.no_grad():
        for batch_idx, (data, partial, target) in enumerate(test_loader):
        #for data, partial, target in test_data:
            #data, target = data.to(device), target.to(device)
            #data = torch.from_numpy(data)
            #partial = torch.from_numpy(partial)
            #target = torch.from_numpy(target)
            
            data, partial, target = data.to(device), partial.to(device), target.to(device)
            #data = torch.
            output = p_net.forward(data)
            s_output = s_net.forward(data, partial, input_x)
            
            pred = output.data.max(1, keepdim=True)[1]
            s_pred = s_output.data.max(1, keepdim=True)[1]
            targ_pred = target.data.max(1, keepdim=True)[1]
            
            pred_list.append(pred)
            s_pred_list.append(s_pred)
            targ_pred_list.append(targ_pred)
    pred = torch.vstack(pred_list)
    s_pred = torch.vstack(s_pred_list)
    targ_pred_list = torch.vstack(targ_pred_list)
    print(pred.shape)
    print(s_pred.shape)
    print(targ_pred.shape)
    return -1

k = 10

datasets = ['Soccer Player','lost','MSRCv2','BirdSong','Yahoo! News']


for filename in datasets:
    for tech in ["sample","select"]:
        for input_x in [False,True]:
            for fold_no in range(k):
              
                train_dataset, val_dataset, test_dataset, input_dim, output_dim = loadTrainAnalysis(filename+".mat", fold_no, k)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                  batch_size=batch_size_train, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_dataset,
                  batch_size=batch_size_test, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=batch_size_test, shuffle=True)
                
                
                p_net = Prediction_Net(input_dim, output_dim)
                s_net = Selection_Net(input_dim, output_dim, input_x)
                
                p_net.to(device)
                s_net.to(device)
            
                
                #model_filename = "results/05012020/"+filename+"/SelectR_"+str(tech)+"_"+str(input_x)+"/models/"+str(fold_no)+"_best.pth"
                model_filename = "results/05012020/"+filename+"/SelectR_"+str(tech)+"_"+str(input_x)+"/models/"+str(fold_no)+"_best.pth"
                
                checkpoint = torch.load(model_filename)
                p_net.load_state_dict(checkpoint['p_net_state_dict'])
                s_net.load_state_dict(checkpoint['s_net_state_dict'])
                
                train_dic = test(train_dataset, input_x)
                val_dic = test(val_dataset, input_x)
                test_dic = test(test_dataset, input_x)
                
    
                #os.makedirs(os.path.dirname(result_filename), exist_ok=True)
                #with open(result_filename,"w", newline='') as file:
                #    file.write(str(train_acc)+"\n")
                #    file.write(str(val_acc)+"\n")
                #    file.write(str(test_acc)+"\n")
                    
                #os.makedirs(os.path.dirname(result_log_filename), exist_ok=True)
                #with open(result_log_filename,"w", newline='') as file:
                #    writer = csv.writer(file)
                #    writer.writerow(["Train Count", "Train Acc", "Test Count", "Test Acc"])
                #    for i in range(len(vals[0])):
                #        writer.writerow([vals[0][i], vals[1][i], vals[2][i], vals[3][i]])