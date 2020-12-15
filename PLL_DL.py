

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
batch_size_train = 2000
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
    #print(output.shape)
    #print(target.shape)
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.log(loss)
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
    


#datasets = ['KMNIST', 'FashionMNIST','MNIST']
#losses = [rl_loss, naive_loss, min_loss, ]

datasets = ['MSRCv2','Yahoo! News','BirdSong','Soccer Player', 'Lost']
losses = [naive_loss, cc_loss, rl_loss,  min_loss]


for filename in datasets:
    
    
    train_dataset, test_dataset, real_train_dataset, input_dim, output_dim = loadTrain(filename+".mat")
    train_loader = torch.utils.data.DataLoader(train_dataset,
      batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
      batch_size=batch_size_test, shuffle=True)
    real_train_loader = torch.utils.data.DataLoader(real_train_dataset,
      batch_size=batch_size_test, shuffle=True)
    #loss = rl_loss
    for loss in losses:
        #network = Net(input_dim, output_dim)
        #network.to(device)
        #vals = [[],[],[],[]]
        
        #optimizer = torch.optim.Adam(network.parameters())
        #network.optimizer = optimizer
        #network.train_loader = real_train_loader
        #network.test_loader = test_loader
        #network.real_train_loader = real_train_loader
        
        #f = open("results/"+filename+"_"+str(loss.__name__)+"_linear.txt","w")
        
        #for epoch in range(1, n_epochs + 1):
        #  network.myTrain(epoch, loss, vals)
        #  network.myTest(loss, vals)
          #print(vals)
       # with open("results/"+filename+"/"+filename+"_"+str(loss.__name__)+"_"+str("PureLabels")+".csv","w", newline='') as file:
       #     writer = csv.writer(file)
       #     writer.writerow(["Train Count", "Train Acc", "Test Count", "Test Acc"])
       #     for i in range(len(vals[0])):
       #         writer.writerow([vals[0][i], vals[1][i], vals[2][i], vals[3][i]])
               
        for trial_no in range(5):
            print(filename)
            network = Net(input_dim, output_dim)
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
            with open("results/"+filename+"/"+filename+"_"+str(loss.__name__)+"_"+str(trial_no)+".csv","w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Train Count", "Train Acc", "Test Count", "Test Acc"])
                for i in range(len(vals[0])):
                    writer.writerow([vals[0][i], vals[1][i], vals[2][i], vals[3][i]])


#small dev for early stopping
#big dev for hyperparameters
















