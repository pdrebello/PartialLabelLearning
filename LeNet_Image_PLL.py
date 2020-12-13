#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:15:23 2020

@author: pratheek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:02:33 2020

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
#import torch_optimizer as optim
import scipy.io
from dataset import Dataset, ConvDataset, loadTrain
import sys
from IPython.core.debugger import Pdb
import random
import csv

n_epochs = 60
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vals = [[],[],[],[]]

def rl_loss(output, target):
    prob = output.detach()
    target_probs = (prob*target.float()).sum(dim=1)
    mask = target == 1
    loss = (prob[mask]*torch.log(output[mask])/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
    return -loss

def naive_loss(output, target):
    #print(output.shape)
    #print(target.shape)
    batch_size = output.shape[0]
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

class LeNet5(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=output_dim),
        )
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.real_train_loader = None


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs
    

    def myTrain(self, epoch, loss_function, vals):
      self.train()
      for batch_idx, (data, target) in enumerate(self.train_loader):
        data, target = data.to(device), target.to(device)
        #print(torch.sum(target,dim=1))
        self.optimizer.zero_grad()
        #print(batch_idx)
        #print(data.shape)
        #print(target.shape)
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


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels]

def make_partials(target, output_dim):
    #options = [1,2,3,4]
    #howmany = random.choice(options)
    for i in target:
        rand = torch.FloatTensor(output_dim).uniform_() > 0.5
        #index_options = list(range(output_dim))
        #indices = random.sample(index_options, howmany)
        i[rand] = 1
    return target
    


datasets = ['KMNIST', 'FashionMNIST','MNIST']
losses = [naive_loss, min_loss, rl_loss]

input_dim = 32*32
output_dim = 10
for filename in datasets:
    
    if(filename == "MNIST"):
        train_loader = torchvision.datasets.MNIST('datasets/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                 torchvision.transforms.ToTensor()]))
        test_loader = torchvision.datasets.MNIST('datasets/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                 torchvision.transforms.ToTensor()]))
    elif(filename == "KMNIST"):
        train_loader = torchvision.datasets.KMNIST('datasets/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                 torchvision.transforms.ToTensor()]))
        test_loader = torchvision.datasets.KMNIST('datasets/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                 torchvision.transforms.ToTensor()]))
    else:
        train_loader = torchvision.datasets.FashionMNIST('datasets/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                 torchvision.transforms.ToTensor()]))
        test_loader = torchvision.datasets.FashionMNIST('datasets/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                 torchvision.transforms.ToTensor()]))
    
    data = []
    labels = []
    for image, label in train_loader:
        data.append(image)
        labels.append(label)
    data = torch.stack(data)
    labels = torch.LongTensor(labels)
    target = one_hot_embedding(labels, output_dim)
    
    
    
    real_target = target.clone()
    make_partials(target, output_dim)
    
    train_dataset = ConvDataset(data.numpy(), target.numpy())
    real_train_dataset = ConvDataset(data.numpy(), real_target.numpy())
    
    real_train_loader = torch.utils.data.DataLoader(real_train_dataset, batch_size=batch_size_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        
    data = []
    labels = []
    for image, label in test_loader:
        data.append(image)
        labels.append(label)
    data = torch.stack(data)
    #print(data.shape)
    labels = torch.LongTensor(labels)
    target = one_hot_embedding(labels, output_dim)
    test_dataset = ConvDataset(data.numpy(), target.numpy())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)
    
    #loss = rl_loss
    for loss in losses:
        network = LeNet5(input_dim, output_dim)
        network.to(device)
        vals = [[],[],[],[]]
        
        optimizer = torch.optim.Adam(network.parameters())
        network.optimizer = optimizer
        network.train_loader = real_train_loader
        network.test_loader = test_loader
        network.real_train_loader = real_train_loader
        
        #f = open("results/"+filename+"_"+str(loss.__name__)+"_linear.txt","w")
        
        for epoch in range(1, n_epochs + 1):
          network.myTrain(epoch, loss, vals)
          network.myTest(loss, vals)
          #print(vals)
        with open("results/"+filename+"/"+filename+"_"+str(loss.__name__)+"_"+str("PureLabels_LeNet")+".csv","w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Train Count", "Train Acc", "Test Count", "Test Acc"])
            for i in range(len(vals[0])):
                writer.writerow([vals[0][i], vals[1][i], vals[2][i], vals[3][i]])
                
        for trial_no in range(1):
            network = LeNet5(input_dim, output_dim)
            network.to(device)
            vals = [[],[],[],[]]
            
            optimizer = torch.optim.Adam(network.parameters())
            network.optimizer = optimizer
            network.train_loader = train_loader
            network.test_loader = test_loader
            network.real_train_loader = real_train_loader
            
            #f = open("results/"+filename+"_"+str(loss.__name__)+"_linear.txt","w")
            
            for epoch in range(1, n_epochs + 1):
              network.myTrain(epoch, loss, vals)
              network.myTest(loss, vals)
              #print(vals)
            with open("results/"+filename+"/"+filename+"_"+str(loss.__name__)+"_"+str(trial_no)+"_LeNet.csv","w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Train Count", "Train Acc", "Test Count", "Test Acc"])
                for i in range(len(vals[0])):
                    writer.writerow([vals[0][i], vals[1][i], vals[2][i], vals[3][i]])
        











