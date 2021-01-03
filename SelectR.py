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
from dataset import Dataset, loadTrain, loadTrainT
import sys
from IPython.core.debugger import Pdb
import random
import csv
import os

n_epochs = 150
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

epsilon = 1e-6

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    #loss = torch.log((a*p + (1-a)*(1-p)).sum(dim=1))
    #
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
    def __init__(self, input_dim, output_dim):
        super(Phi_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x, targetSet, rl_technique):
        mask = targetSet>0
        
        x = x * targetSet
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
    def __init__(self, input_dim, output_dim):
        super(Selection_Net, self).__init__()
        
        self.p_net = Prediction_Net(input_dim, output_dim)
        for param in self.p_net.parameters():
            param.requires_grad = False
            
        self.phi_net = Phi_Net(output_dim, output_dim)

    def forward(self, x, targetSet, rl_technique):
        with torch.no_grad():
            x = self.p_net(x)
        
        x = self.phi_net(x, targetSet, rl_technique)
        return x

def pre_train(epoch):
    p_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        p_optimizer.zero_grad()
        output = p_net(data)
        
        
        loss = cc_loss(output, target)
        loss.backward()
        
        p_optimizer.step()
        
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    correct = 0
    with torch.no_grad():
        for data, target in real_train_loader:
            data, target = data.to(device), target.to(device)
            output = p_net.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            targ_pred = target.data.max(1, keepdim=True)[1]
            correct += pred.eq(targ_pred.data.view_as(pred)).sum()
  
        print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        0, correct, len(real_train_loader.dataset),
        100. * correct / len(real_train_loader.dataset)))


def train(epoch, rl_technique):
    p_net.train()
    s_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        p_optimizer.zero_grad()
        s_optimizer.zero_grad()
        
        #prob output of prediction network
        p = p_net(data)
        q = s_net(data, target, rl_technique)
        
        #Pdb().set_trace()
        if(rl_technique == 'sample'):
            if(torch.isnan(q.sum())):
                Pdb().set_trace()
            with torch.no_grad():
                m = torch.distributions.bernoulli.Bernoulli(q)
                a = m.sample()
            mask = target
            reward = sample_reward_function(p.detach(), q, a, mask)
            loss = sample_loss_function(p, a)
        else:
            loss = select_loss_function(p, q.detach())
            reward = select_reward_function(p.detach(), q)
        if(torch.isnan(loss)):
            Pdb().set_trace()
        if(torch.isinf(loss)):
            Pdb().set_trace()
        loss.backward()
        reward.backward()
        
        p_optimizer.step()
        s_optimizer.step()
        
        
        if batch_idx % 100 == 0:
            s_net.p_net.copy(p_net)
        
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tReward: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), reward.item()))
          
    correct = 0
    with torch.no_grad():
        for data, target in real_train_loader:
            data, target = data.to(device), target.to(device)
            output = p_net.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            targ_pred = target.data.max(1, keepdim=True)[1]
            correct += pred.eq(targ_pred.data.view_as(pred)).sum()
  
        print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        0, correct, len(real_train_loader.dataset),
        100. * correct / len(real_train_loader.dataset)))
        
        vals[0].append(correct.item())
        vals[1].append(100. * float(correct.item()) / len(real_train_loader.dataset))

def test(test_data):
    p_net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = p_net.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            targ_pred = target.data.max(1, keepdim=True)[1]
            correct += pred.eq(targ_pred.data.view_as(pred)).sum()
      
      
    test_loss /= len(test_data.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset)))
    vals[2].append(correct.item())
    vals[3].append(100. * float(correct.item()) / len(test_data.dataset))
    return (100. * float(correct.item()) / len(test_data.dataset))
    

k = 10

datasets = ['Soccer Player','MSRCv2','BirdSong','Yahoo! News','lost',]

for tech in ["sample","select"]:
    for filename in datasets:
        for fold_no in range(k):
            
            train_dataset, test_dataset, real_train_dataset, val_dataset, input_dim, output_dim = loadTrain(filename+".mat", fold_no, k)
            train_loader = torch.utils.data.DataLoader(train_dataset,
              batch_size=batch_size_train, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset,
              batch_size=batch_size_test, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset,
              batch_size=batch_size_test, shuffle=True)
            real_train_loader = torch.utils.data.DataLoader(real_train_dataset,
              batch_size=batch_size_train, shuffle=True)
            
            vals = [[],[],[],[]]
            
            p_net = Prediction_Net(input_dim, output_dim)
            s_net = Selection_Net(input_dim, output_dim)
            
            p_net.to(device)
            s_net.to(device)
            
            p_optimizer = torch.optim.Adam(p_net.parameters())
            s_optimizer = torch.optim.Adam(s_net.parameters())
        
        
            best_val = 0
            result_filename = "results/"+filename+"/"+str("SelectR")+"/results/"+str(fold_no)+"_out.txt"
            result_log_filename = "results/"+filename+"/"+str("SelectR")+"/logs/"+str(fold_no)+"_log.csv"
            model_filename = "results/"+filename+"/"+str("SelectR")+"/models/"+str(fold_no)+"_best.pth"
            
            
            load_pre_train = "results/"+filename+"/"+str("cc_loss")+"/models/"+str(fold_no)+"_10.pth"
            p_net.load_state_dict(torch.load(load_pre_train))
            s_net.p_net.copy(p_net)
            
            #for epoch in range(1, 1):
            #    pre_train(epoch)
            #    test()
            #s_net.p_net.copy(p_net)
        
        
            for epoch in range(1, n_epochs + 1):
              train(epoch, tech)
              val = test(val_loader)
              
              if(val > best_val):
                  best_val = val
                  os.makedirs(os.path.dirname(model_filename), exist_ok=True)
                  torch.save(p_net.state_dict(), model_filename)
              if((epoch%10==0) and (epoch>0)):
                  e_model_filename = "results/"+filename+"/SelectR/models/"+str(fold_no)+"_"+str(epoch)+".pth"
                  os.makedirs(os.path.dirname(e_model_filename), exist_ok=True)
                  torch.save(p_net.state_dict(), e_model_filename)
            
            
            p_net.load_state_dict(torch.load(model_filename))
            train_acc = test(real_train_loader)
            val_acc = test(val_loader)
            test_acc = test(test_loader)
            
            os.makedirs(os.path.dirname(result_filename), exist_ok=True)
            with open(result_filename,"w", newline='') as file:
                file.write(str(train_acc)+"\n")
                file.write(str(val_acc)+"\n")
                file.write(str(test_acc)+"\n")
                
            os.makedirs(os.path.dirname(result_log_filename), exist_ok=True)
            with open(result_log_filename,"w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Train Count", "Train Acc", "Test Count", "Test Acc"])
                for i in range(len(vals[0])):
                    writer.writerow([vals[0][i], vals[1][i], vals[2][i], vals[3][i]])