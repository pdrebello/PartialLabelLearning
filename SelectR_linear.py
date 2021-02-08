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

import scipy.io
from dataset import Dataset, loadTrain, loadTrainT
import sys
from IPython.core.debugger import Pdb
import random
import csv
import os
import json
import argparse

parser = argparse.ArgumentParser(description = "Description for my parser")

parser.add_argument('--dataset', type=str, help="dataset")
parser.add_argument('--fold_no', type=int, help="fold number")
parser.add_argument('--dump_dir', type=str, help="dump directory for results")
parser.add_argument('--pretrain', type=str, help="training procedure")

argument = parser.parse_args()
   

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
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def select_loss_function(p, q, mask):
    loss = (q*mask*torch.log(p+epsilon)).sum(dim=1).mean()
    return -loss

def select_reward_function(p, q, mask):
    rew = (q*p*mask).sum(dim=1).mean()
    return -rew

def sample_reward_function(p, q, a, mask):
    
    rew = (a*p).sum(dim=1) - ((1-a)*mask*p).sum(dim=1)
    lap = a*torch.log(q+ epsilon) + (1-a)*mask*torch.log(1-q+ epsilon)
    lap = lap.sum(dim=1)
    
    return -torch.mean(rew * lap)

def sample_loss_function(p, q, a, mask):
    loss = torch.log((a*p*mask).sum(dim=1) +epsilon)
    return -0.5*torch.mean(loss)

class Prediction_Net_Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Prediction_Net_Linear, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform(self.fc1.weight)

    def forward(self, x):
        x = F.softmax(self.fc1(x))
        return x
    
    def copy(self, net2):
        self.load_state_dict(net2.state_dict())

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

def pre_train(epoch, p_net, p_optimizer):
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
    
    return 100. * float(correct.item()) / len(real_train_loader.dataset)

def train(epoch, rl_technique, p_net, p_optimizer, s_net, s_optimizer):
    p_net.train()
    s_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        p_optimizer.zero_grad()
        s_optimizer.zero_grad()
        
        #prob output of prediction network
        p = p_net(data)
        q = s_net(data, target, rl_technique)
        
        if(rl_technique == 'sample'):
            if(torch.isnan(q.sum())):
                Pdb().set_trace()
            with torch.no_grad():
                m = torch.distributions.bernoulli.Bernoulli(q)
                a = m.sample()
            mask = target
            reward = sample_reward_function(p.detach(), q, a, mask)
            loss = sample_loss_function(p, q.detach(), a, mask)
        else:
            mask = target
            loss = select_loss_function(p, q.detach(),mask)
            reward = select_reward_function(p.detach(), q, mask)
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
    return 100. * float(correct.item()) / len(real_train_loader.dataset)

def test(test_data, p_net):
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

def q_test(test_data, q_net, rl_technique):
    q_net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            
            data, target = data.to(device), target.to(device)
            one_hot = torch.ones_like(target)
            one_hot.to(device)
            output = q_net.forward(data, one_hot, rl_technique)
            pred = output.data.max(1, keepdim=True)[1]
            targ_pred = target.data.max(1, keepdim=True)[1]
            correct += pred.eq(targ_pred.data.view_as(pred)).sum()
      
      
    test_loss /= len(test_data.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset)))
    #vals[2].append(correct.item())
    #vals[3].append(100. * float(correct.item()) / len(test_data.dataset))
    return (100. * float(correct.item()) / len(test_data.dataset))
   
def save_checkpoint(epoch, val_acc, p_net, p_optimizer, s_net, s_optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'val_acc': val,
        'p_net_state_dict': p_net.state_dict(),
        'p_optimizer': p_optimizer.state_dict(),
        's_net_state_dict': s_net.phi_net.state_dict(),
        's_optimizer': s_optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)
 
#dump_dir = os.path.join("results", "01022021")
dump_dir = argument.dump_dir
filename = argument.dataset
fold_no = argument.fold_no
pretrain = argument.pretrain
k = 10

#datasets = ['lost','MSRCv2','BirdSong','Yahoo! News','Soccer Player']


if(filename == 'lost'):
    n_epochs = 1000
else:
    n_epochs = 500
    
p_pretrain_epochs = 10
q_pretrain_epochs = 50

#n_epochs = 2
#p_pretrain_epochs = 2
#q_pretrain_epochs = 2




for tech in ["sample","select"]:
    for input_x in [True, False]:  
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
        
        p_net_linear = Prediction_Net_Linear(input_dim, output_dim)
        p_net_mlp = Prediction_Net(input_dim, output_dim)
        s_net = Selection_Net(input_dim, output_dim, input_x)
        
        p_net_linear.to(device)
        p_net_mlp.to(device)
        s_net.to(device)
        
        p_optimizer_linear = torch.optim.Adam(p_net_linear.parameters())
        p_optimizer_mlp = torch.optim.Adam(p_net_mlp.parameters())
        s_optimizer = torch.optim.Adam(s_net.parameters())
    
    
        best_val = 0
        best_val_epoch = -1

        dataset_technique_path = os.path.join(filename, pretrain, "SelectR_" + str(tech) + "_" + str(input_x))
        
        #result_filename = "results/31012020/"+filename+"/SelectR_"+str(tech)+"_"+str(input_x)+"/results/"+str(fold_no)+"_out.txt"
        #result_log_filename = "results/31012020/"+filename+"/SelectR_"+str(tech)+"_"+str(input_x)+"/logs/"+str(fold_no)+"_log.csv"
        #model_filename = "results/31012020/"+filename+"/SelectR_"+str(tech)+"_"+str(input_x)+"/models/"+str(fold_no)+"_best.pth"
        
        result_filename = os.path.join(dump_dir, dataset_technique_path, "results", str(fold_no)+"_out.txt")
        result_log_filename = os.path.join(dump_dir, dataset_technique_path, "logs", str(fold_no)+"_log.csv")
        result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", str(fold_no)+"_log.json")
        
        
        
        p_linear_pre_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "p_linear_pre_train", str(fold_no)+".pth")
        q_pre_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "q_pre_train", str(fold_no)+".pth")
        p_mlp_pre_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "p_mlp_pre_train", str(fold_no)+".pth")
        train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train", str(fold_no)+".pth") 

        logs = []
        
        if(pretrain == 'pretrain_q'):
            s_net.p_net = Prediction_Net_Linear(input_dim, output_dim)
            
            for epoch in range(1,p_pretrain_epochs+1):
                train_acc = pre_train(epoch, p_net_linear, p_optimizer_linear)
                val = test(val_loader, p_net_linear)
                
                log = {'epoch':epoch, 'phase': 'p_linear_pre_train', 'train_acc': train_acc, 'val_acc': val}
                logs.append(log)
                
                
            save_checkpoint(p_pretrain_epochs, val, p_net_linear, p_optimizer_linear, s_net, s_optimizer, p_linear_pre_train_checkpoint)
    
            for param in s_net.p_net.parameters():
                param.requires_grad = False
            s_net.to(device)   
            for epoch in range(1,q_pretrain_epochs+1):
                train_acc = train(epoch, tech, p_net_linear, p_optimizer_linear, s_net, s_optimizer)
                val = test(val_loader, p_net_linear)
                q_val = q_test(val_loader, s_net, tech)
                
                log = {'epoch':epoch, 'phase': 'q_pre_train', 'train_acc': train_acc, 'val_acc': val, 'q_val_acc':q_val}
                logs.append(log)
                
                
            save_checkpoint(q_pretrain_epochs, val, p_net_linear, p_optimizer_linear, s_net, s_optimizer, q_pre_train_checkpoint)

        s_net.p_net = Prediction_Net(input_dim, output_dim)
        for param in s_net.p_net.parameters():
            param.requires_grad = False
        
        if(pretrain != 'without_pretrain_p'):
            for epoch in range(1,p_pretrain_epochs+1):
                train_acc = pre_train(epoch, p_net_mlp, p_optimizer_mlp)
                val = test(val_loader, p_net_mlp)
                
                log = {'epoch':epoch, 'phase': 'p_mlp_pre_train', 'train_acc': train_acc, 'val_acc': val}
                logs.append(log)
                
            save_checkpoint(p_pretrain_epochs, val, p_net_mlp, p_optimizer_mlp, s_net, s_optimizer, p_mlp_pre_train_checkpoint)
        
        s_net.to(device)
        for epoch in range(1, n_epochs+1):
            train_acc = train(epoch, tech, p_net_mlp, p_optimizer_mlp, s_net, s_optimizer)
            val = test(val_loader, p_net_mlp)
            q_val = q_test(val_loader, s_net, tech)
              
            log = {'epoch':epoch, 'phase': 'train', 'train_acc': train_acc, 'val_acc': val, 'q_val_acc':q_val}
            logs.append(log)
            
            
            if(val > best_val):
                best_val = val
                best_val_epoch = epoch
                save_checkpoint(epoch, val, p_net_mlp, p_optimizer_mlp, s_net, s_optimizer, train_checkpoint)
                            

       
        
        checkpoint = torch.load(train_checkpoint)
        p_net_mlp.load_state_dict(checkpoint['p_net_state_dict'])
        train_acc = test(real_train_loader, p_net_mlp)
        val_acc = test(val_loader, p_net_mlp)
        test_acc = test(test_loader, p_net_mlp)
        q_test_acc = q_test(test_loader, s_net, tech)
        
        log = {'epoch':-1, 'train_epoch': best_val_epoch, 'phase': 'test', 'train_acc': train_acc, 'val_acc': val, 'test_acc': test_acc, 'q_test_acc': q_test_acc}
        logs.append(log)
        
        os.makedirs(os.path.dirname(result_filename), exist_ok=True)
        with open(result_filename,"w", newline='') as file:
            file.write(str(train_acc)+"\n")
            file.write(str(val_acc)+"\n")
            file.write(str(test_acc)+"\n")
            
        os.makedirs(os.path.dirname(result_log_filename), exist_ok=True)
        
        os.makedirs(os.path.dirname(result_log_filename_json), exist_ok=True)
        with open(result_log_filename_json, "w") as file:
            for log in logs:
                json.dump(log, file)
                file.write("\n")
        
        with open(result_log_filename,"w", newline='') as file:
            
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Acc", "Test Count", "Test Acc"])
            for i in range(len(vals[0])):
                writer.writerow([vals[0][i], vals[1][i], vals[2][i], vals[3][i]])
