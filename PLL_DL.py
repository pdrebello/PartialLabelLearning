

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
#import torch_optimizer as optim
import scipy.io
from dataset import Dataset, loadTrain
import sys
from IPython.core.debugger import Pdb
import random
import csv

n_epochs = 150

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

def rl_loss(output, target):
    prob = output.detach()
    target_probs = (prob*target.float()).sum(dim=1)
    mask = ((target == 1) & (prob > epsilon))
    loss = (prob[mask]*torch.log(output[mask])/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
    return -loss

def cc_loss(output, target):
    
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.log(loss)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def cour_loss(output, target):
    batch_size = output.shape[0]
    comp_output = (1 - output)/(output.shape[1]-1)
    comp_target = 1 - target
    comp_loss = torch.bmm(comp_output.view(comp_output.shape[0], 1, comp_output.shape[1]), comp_target.view(comp_output.shape[0], comp_output.shape[1], 1))
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.log(loss+comp_loss)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def naive_loss(output, target):
    
    batch_size = output.shape[0]
    loss = torch.log(output + epsilon)
    normalize = torch.sum(target, dim = 1)
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1)).flatten()
    loss = loss/normalize
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def min_loss(output, target):
    batch_size = output.shape[0]
    loss = output * target
    #res = loss <= 0
    #loss[res] = -10000
    loss = torch.max(loss, dim = 1).values
    loss = torch.log(loss)
    
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        #self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.softmax(self.fc3(x))
        #x = F.softmax(self.fc1(x))
        return x

    def myTrain(self, epoch, loss_function, vals):
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
      
    
    def myTest(self, loss_function, vals, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.forward(data)
                pred = output.data.max(1, keepdim=True)[1]
                targ_pred = target.data.max(1, keepdim=True)[1]
                correct += pred.eq(targ_pred.data.view_as(pred)).sum()
          
          
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        
        vals[2].append(correct.item())
        vals[3].append(100. * float(correct.item()) / len(test_loader.dataset))
        return (100. * float(correct.item()) / len(test_loader.dataset))

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels]

def make_partials(target, output_dim):
    for i in target:
        rand = torch.FloatTensor(output_dim).uniform_() > 0.5
        i[rand] = 1
    return target
    

k = 10
datasets = ['Soccer Player','MSRCv2','lost','BirdSong','Yahoo! News']
losses = [cc_loss, naive_loss, rl_loss,  min_loss]


for filename in datasets:
    if(filename == 'lost'):
        batch_size_train = 4
    else:
        batch_size_train = 64
    for loss in losses:
    
        for fold_no in range(k):
            
            train_dataset, test_dataset, real_train_dataset, val_dataset, input_dim, output_dim = loadTrain(filename+".mat", fold_no, k)
            
            train_loader = torch.utils.data.DataLoader(train_dataset,
              batch_size=batch_size_train, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset,
              batch_size=batch_size_test, shuffle=True)
            real_train_loader = torch.utils.data.DataLoader(real_train_dataset,
              batch_size=batch_size_train, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset,
              batch_size=batch_size_train, shuffle=True)
            
            network = Net(input_dim, output_dim)
            network.to(device)
            vals = [[],[],[],[]]
            
            optimizer = torch.optim.Adam(network.parameters())
            network.optimizer = optimizer
            network.train_loader = train_loader
            network.test_loader = test_loader
            network.real_train_loader = real_train_loader
            network.val_loader = val_loader
            
            best_val = 0
            result_filename = "results/"+filename+"/"+str(loss.__name__)+"/results/"+str(fold_no)+"_out.txt"
            result_log_filename = "results/"+filename+"/"+str(loss.__name__)+"/logs/"+str(fold_no)+"_log.csv"
            model_filename = "results/"+filename+"/"+str(loss.__name__)+"/models/"+str(fold_no)+"_best.pth"
               
            for epoch in range(1, n_epochs + 1):
              network.myTrain(epoch, loss, vals)
              val = network.myTest(loss, vals, val_loader)
                 
              if(val > best_val):
                  best_val = val
                  os.makedirs(os.path.dirname(model_filename), exist_ok=True)
                  torch.save(network.state_dict(), model_filename)
              if((epoch%10==0) and (epoch>0)):
                  e_model_filename = "results/"+filename+"/"+str(loss.__name__)+"/models/"+str(fold_no)+"_"+str(epoch)+".pth"
                  os.makedirs(os.path.dirname(e_model_filename), exist_ok=True)
                  torch.save(network.state_dict(), e_model_filename)
            
            
            network.load_state_dict(torch.load(model_filename))
            train_acc = network.myTest(loss, vals, real_train_loader)
            val_acc = network.myTest(loss, vals, val_loader)
            test_acc = network.myTest(loss, vals, test_loader)
            
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


















