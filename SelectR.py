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
from dataset import Dataset, loadTrain
import sys
from IPython.core.debugger import Pdb
import random
import csv

n_epochs = 300
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
    #print(output.shape)
    #print(target.shape)
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.log(loss)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def naive_reward(output, target):
    #print(output.shape)
    #print(target.shape)
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    #loss = torch.log(loss)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

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

class Selection_Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Selection_Net, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        self.fc1.requires_grad=False
        self.fc2.requires_grad=False
        self.fc3.requires_grad=False
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        
        self.fc1_g = nn.Linear(output_dim*2, 512)
        self.fc2_g = nn.Linear(512, 256)
        self.fc3_g = nn.Linear(256, output_dim)
        torch.nn.init.xavier_uniform(self.fc1_g.weight)
        torch.nn.init.xavier_uniform(self.fc2_g.weight)
        torch.nn.init.xavier_uniform(self.fc3_g.weight)
        self.bn1_g = nn.BatchNorm1d(512)
        self.bn2_g = nn.BatchNorm1d(256)
        
        #self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x, targetSet):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.softmax(self.fc3(x))
        
        #print(x.shape)
        x_i = x.argmax(dim = 1)
        j = torch.arange(x.size(0)).long()
        #print(x_i)
        x = torch.zeros(x.shape)
        x[j, x_i] = 1
        
        #print(x.shape)
        #print(targetSet.shape)
        inp2 = torch.cat((x, targetSet), dim=1)
        #print(inp2.shape)
        inp2 = F.elu(self.bn1_g(self.fc1_g(inp2)))
        inp2 = F.elu(self.bn2_g(self.fc2_g(inp2)))
        
        #Pdb().set_trace()
        inp2 = F.softmax(self.fc3_g(inp2))*targetSet
        #print(inp2.shape)
        #print(inp2.sum(dim=1, keepdim=True)[1].shape)
        inp2 = inp2/inp2.sum(dim=1, keepdim=True)[1]
        
        
        #x = F.softmax(self.fc1(x))
        return x, inp2




filename = "Soccer Player"
train_dataset, test_dataset, real_train_dataset, input_dim, output_dim = loadTrain(filename+".mat")
train_loader = torch.utils.data.DataLoader(train_dataset,
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
  batch_size=batch_size_test, shuffle=True)
real_train_loader = torch.utils.data.DataLoader(real_train_dataset,
  batch_size=batch_size_train, shuffle=True)

loss_function = cc_loss
reward_function = naive_reward

p_net = Prediction_Net(input_dim, output_dim)
s_net = Selection_Net(input_dim, output_dim)

p_optimizer = torch.optim.Adam(p_net.parameters())
s_optimizer = torch.optim.Adam(s_net.parameters())

def train(epoch):
    p_net.train()
    s_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        p_optimizer.zero_grad()
        #s_optimizer.zero_grad()
        
        output = p_net(data)
        
        yhat, probs = s_net(data, target)
        #print(probs.shape)
        m = torch.distributions.Categorical(probs)
        
        target_choice = m.sample()
        #print(target_choice)
        j = torch.arange(probs.size(0)).long()
        rl_target = torch.zeros(target.shape)
        rl_target[j, target_choice] = 1
        loss = loss_function(output, rl_target)
        
        
        loss = loss_function(output, rl_target)
        reward = reward_function(probs, yhat)
        
        loss.backward()
        #p_optimizer.step()
        reward.backward()
        p_optimizer.step()
        
        s_net.fc1.weight.data = p_net.fc1.weight.data
        s_net.fc1.bias.data = p_net.fc1.bias.data
        s_net.fc2.weight.data = p_net.fc2.weight.data
        s_net.fc2.bias.data = p_net.fc2.bias.data
        s_net.fc3.weight.data = p_net.fc3.weight.data
        s_net.fc3.bias.data = p_net.fc3.bias.data
        #s_optimizer.step()
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

def test():
    p_net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = p_net.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            targ_pred = target.data.max(1, keepdim=True)[1]
            correct += pred.eq(targ_pred.data.view_as(pred)).sum()
      
      
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    #f.write()
    #f.write('Test: {}/{} ({:.0f}%)\n'.format(
    #        test_loss, correct, len(self.test_loader.dataset),
    #        100. * correct / len(self.test_loader.dataset)))
    #f.write("\n")

  #f.write('Train: {}/{} ({:.0f}%)\n'.format(correct, len(self.real_train_loader.dataset),
  #  100. * correct / len(self.real_train_loader.dataset)))
  #f.write("\n")
  #vals[0].append(correct.item())
  #vals[1].append(100. * float(correct.item()) / len(self.real_train_loader.dataset))
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
"""
for batch_idx, (data, target) in enumerate(self.train_loader):
    data, target = data.to(device), target.to(device)
    #print(torch.sum(target,dim=1))
    self.optimizer.zero_grad()
    output = network(data)
  
self.train()
for batch_idx, (data, target) in enumerate(self.train_loader):
    data, target = data.to(device), target.to(device)
    #print(torch.sum(target,dim=1))
    self.optimizer.zero_grad()
    output = network(data)
    
    loss = loss_function(output, target)
    
    loss.backward()
    self.optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(self.train_loader.dataset),
        100. * batch_idx / len(self.train_loader), loss.item()))
  correct = 0
  with torch.no_grad():
      for data, target in self.real_train_loader:
          data, target = data.to(device), target.to(device)
          output = self.forward(data)
          pred = output.data.max(1, keepdim=True)[1]
          targ_pred = target.data.max(1, keepdim=True)[1]
          correct += pred.eq(targ_pred.data.view_as(pred)).sum()
      
  print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    0, correct, len(self.real_train_loader.dataset),
    100. * correct / len(self.real_train_loader.dataset)))
  #f.write('Train: {}/{} ({:.0f}%)\n'.format(correct, len(self.real_train_loader.dataset),
  #  100. * correct / len(self.real_train_loader.dataset)))
  #f.write("\n")
  vals[0].append(correct.item())
  vals[1].append(100. * float(correct.item()) / len(self.real_train_loader.dataset))

def myTest(self, loss_function, vals):
    self.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in self.test_loader:
            data, target = data.to(device), target.to(device)
            output = self.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            targ_pred = target.data.max(1, keepdim=True)[1]
            correct += pred.eq(targ_pred.data.view_as(pred)).sum()
      
      
    test_loss /= len(self.test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
    #f.write()
    #f.write('Test: {}/{} ({:.0f}%)\n'.format(
    #        test_loss, correct, len(self.test_loader.dataset),
    #        100. * correct / len(self.test_loader.dataset)))
    #f.write("\n")
    vals[2].append(correct.item())
    vals[3].append(100. * float(correct.item()) / len(self.test_loader.dataset))    

"""  
    
