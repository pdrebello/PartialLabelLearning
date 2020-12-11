#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:05:48 2020

@author: pratheek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:32:26 2020

@author: pratheek
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
import scipy.io
from dataset import Dataset, loadTrain, loadTrainTV
import sys

n_epochs = 1000
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.0001
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_dataset, test_dataset, real_train_dataset, input_dim, output_dim = loadTrainTV("tv_data/lost_with_screenplay_supervision.mat")
train_loader = torch.utils.data.DataLoader(train_dataset,
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
  batch_size=batch_size_test, shuffle=True)
real_train_loader = torch.utils.data.DataLoader(real_train_dataset,
  batch_size=batch_size_test, shuffle=True)

#print(train_dataset.__getitem__(0).shape)

def rl_loss(output, target):
    
    batch_size = output.shape[0]
    #print(output)
    #print(target)
    loss = output * target
    probab = loss
    loss = torch.log(loss)
    loss = (probab * loss)/torch.sum(probab, dim=1)
    
    loss = torch.sum(loss)
    
    loss = torch.div(loss, -batch_size)
    
    return loss

def naive_loss(output, target):
    
    batch_size = output.shape[0]
    #print(output)
    #print(target)
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    
    loss = torch.log(loss)
    
    loss = torch.sum(loss)
    
    loss = torch.div(loss, -batch_size)
    
    return loss

def min_loss(output, target):
    batch_size = output.shape[0]
    loss = output * target
    res = loss <= 0
    loss[res] = 10000
    loss = torch.min(loss, dim = 1).values
    loss = torch.log(loss)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(960, 50)
        #self.fc1 = nn.Linear(16200, 50)
        self.fc2 = nn.Linear(50, 16)

    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        #x = 
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)
    
network = Net()
yogi = optim.Yogi(
    network.parameters(),
    lr= 0.01,
    betas=(0.9, 0.999),
    eps=1e-3,
    initial_accumulator=1e-6,
    weight_decay=0,
)
optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch, loss_function):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    
    loss = loss_function(output, target)
    
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')
  correct = 0
  with torch.no_grad():
    for data, target in real_train_loader:
    
      output = network(data)
      
      pred = output.data.max(1, keepdim=True)[1]
      targ_pred = target.data.max(1, keepdim=True)[1]
      correct += pred.eq(targ_pred.data.view_as(pred)).sum()
      
  print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    0, correct, len(real_train_loader.dataset),
    100. * correct / len(real_train_loader.dataset)))

def test(loss_function):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
    
      output = network(data)
      
      test_loss += loss_function(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      targ_pred = target.data.max(1, keepdim=True)[1]
      
      
      correct += pred.eq(targ_pred.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

#test()
for epoch in range(1, n_epochs + 1):
  train(epoch, rl_loss)
  test(rl_loss)












