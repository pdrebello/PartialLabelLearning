import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import scipy.io
from dataset import Dataset, loadTrain, loadTrainT
from losses import cc_loss, min_loss, naive_loss, iexplr_loss, sample_loss_function, sample_reward_function, select_loss_function, select_reward_function
from networks import Prediction_Net, Prediction_Net_Linear, Selection_Net, Phi_Net
import sys
from IPython.core.debugger import Pdb
import random
import csv
import os
import json
import argparse

parser = argparse.ArgumentParser(description = "Description for my parser")

parser.add_argument('--dataset', type=str, help="dataset")
parser.add_argument('--datasets', type=str, help="list of datasets")
parser.add_argument('--fold_no', type=int, help="fold number")
parser.add_argument('--dump_dir', type=str, help="dump directory for results")
parser.add_argument('--technique', type=str, help="training procedure")
parser.add_argument('--model', type=str, help="Use a 1 layer model for prediction?")
parser.add_argument('--pretrain_stategy', type=str, help="If RL, how to pretrain?")

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


def train(epoch, loss_function, p_net, p_optimizer):
    p_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        p_optimizer.zero_grad()
        output = p_net(data)
        
        loss = loss_function(output, target)
        loss.backward()
        
        p_optimizer.step()
        
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def rl_train(epoch, rl_technique, p_net, p_optimizer, s_net, s_optimizer):
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
          
def p_accuracy(test_data, p_net):
    p_net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = p_net.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += torch.gather(target, 1, pred).sum()
            
    return (100. * float(correct.item()) / len(test_data.dataset))

def q_accuracy(test_data, q_net, rl_technique):
    q_net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_data:

            data, target = data.to(device), target.to(device)
            one_hot = torch.ones_like(target)
            one_hot.to(device)
            output = q_net.forward(data, one_hot, rl_technique)
            pred = output.data.max(1, keepdim=True)[1]
            correct += torch.gather(target, 1, pred).sum()
    return (100. * float(correct) / len(test_data.dataset))

def save_checkpoint(epoch, val_acc, p_net, p_optimizer, s_net, s_optimizer, filename):
    if(s_net is None):
        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'p_net_state_dict': p_net.state_dict(),
            'p_optimizer': p_optimizer.state_dict(),
            's_net_state_dict': None,
            's_optimizer': None,
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'p_net_state_dict': p_net.state_dict(),
            'p_optimizer': p_optimizer.state_dict(),
            's_net_state_dict': s_net.phi_net.state_dict(),
            's_optimizer': s_optimizer.state_dict(),
        }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)


dump_dir = argument.dump_dir
filename = argument.dataset
datasets = argument.datasets
datasets = [str(item) for item in datasets.split(',')]
fold_no = argument.fold_no
technique = argument.technique
model = argument.model
if(model is None):
    model = '3layer'
k = 10


for filename in datasets:
    if(filename in ['lost','MSRCv2','BirdSong']):
        n_epochs = 1000
    else:
        n_epochs = 150
    n_epochs = 2
    train_dataset, real_train_dataset, val_dataset, real_val_dataset, test_dataset, real_test_dataset, input_dim, output_dim = loadTrain(filename+".mat", fold_no, k)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
      batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
      batch_size=batch_size_test, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
      batch_size=batch_size_test, shuffle=True)
    
    real_train_loader = torch.utils.data.DataLoader(real_train_dataset,
      batch_size=batch_size_train, shuffle=True)
    real_test_loader = torch.utils.data.DataLoader(test_dataset,
      batch_size=batch_size_test, shuffle=True)
    real_val_loader = torch.utils.data.DataLoader(val_dataset,
      batch_size=batch_size_test, shuffle=True)
    
    logs = []
    
    if((technique == "cc_loss") or (technique == "min_loss") or (technique == "naive_loss") or (technique == "iexplr_loss")):
        if(technique == "cc_loss"):
            loss_function = cc_loss
        elif(technique == "min_loss"):
            loss_function = min_loss
        elif(technique == "naive_loss"):
            loss_function = naive_loss
        elif(technique == "iexplr_loss"):
            loss_function = iexplr_loss
        
        if(model == "1layer"):
            p_net = Prediction_Net_Linear(input_dim, output_dim)
        else:
            p_net = Prediction_Net(input_dim, output_dim)
            
        p_net.to(device)
        p_optimizer = torch.optim.Adam(p_net.parameters())
        
        dataset_technique_path = os.path.join(filename, model, technique, str(fold_no))
        
        result_filename = os.path.join(dump_dir, dataset_technique_path, "results", "out.txt")
        result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", "log.json")
        train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best.pth") 
        
        best_val = 0
        best_val_epoch = -1
        for epoch in range(1,n_epochs+1):
            train(epoch, loss_function, p_net, p_optimizer)
            surrogate_train_acc = p_accuracy(train_loader, p_net)
            real_train_acc = p_accuracy(real_train_loader, p_net)
            surrogate_val_acc = p_accuracy(val_loader, p_net)
            real_val_acc = p_accuracy(real_val_loader, p_net)
            
            
            log = {'epoch':epoch, 'best_epoch': best_val_epoch,'phase': 'train', 
                       'surrogate_train_acc': surrogate_train_acc, 'real_train_acc': real_train_acc, 
                       'surrogate_val_acc': surrogate_val_acc, 'real_val_acc': real_val_acc, 
                       'surrogate_test_acc': None, 'real_test_acc': None, 
                       'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                       'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                       'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                       'info': dataset_technique_path}
            logs.append(log)
            if(surrogate_val_acc > best_val):
                best_val = surrogate_val_acc
                best_val_epoch = epoch
                save_checkpoint(epoch, surrogate_val_acc, p_net, p_optimizer, None, None, train_checkpoint)
        
        checkpoint = torch.load(train_checkpoint)
        p_net.load_state_dict(checkpoint['p_net_state_dict'])
        surrogate_train_acc = p_accuracy(train_loader, p_net)
        real_train_acc = p_accuracy(real_train_loader, p_net)
        surrogate_val_acc = p_accuracy(val_loader, p_net)
        real_val_acc = p_accuracy(real_val_loader, p_net)
        surrogate_test_acc = p_accuracy(test_loader, p_net)
        real_test_acc = p_accuracy(real_test_loader, p_net)
        
        
        log = {'epoch':-1, 'best_epoch': best_val_epoch, 'phase': 'train', 
                       'surrogate_train_acc': surrogate_train_acc, 'real_train_acc': real_train_acc, 
                       'surrogate_val_acc': surrogate_val_acc, 'real_val_acc': real_val_acc, 
                       'surrogate_test_acc': surrogate_test_acc, 'real_test_acc': real_test_acc, 
                       'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                       'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                       'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                       'info': dataset_technique_path}
        logs.append(log)
        
        os.makedirs(os.path.dirname(result_log_filename_json), exist_ok=True)
        with open(result_log_filename_json, "w") as file:
            for log in logs:
                json.dump(log, file)
                file.write("\n")
    """    
    elif((technique == "linear_rl") or (technique == "exponential_rl")):    
        
        p_net_mlp = Prediction_Net(input_dim, output_dim)
        s_net = Selection_Net(input_dim, output_dim, True)
        
        
        p_net_mlp.to(device)
        s_net.to(device)
        
        p_optimizer_linear = torch.optim.Adam(p_net_linear.parameters())
        p_optimizer_mlp = torch.optim.Adam(p_net_mlp.parameters())
        s_optimizer = torch.optim.Adam(s_net.parameters())
    
        best_val = 0
        best_val_epoch = -1
    
        dataset_technique_path = os.path.join(filename, model, technique, pretrain_stategy, str(fold_no))
        
        
        result_filename = os.path.join(dump_dir, dataset_technique_path, "results", str(fold_no)+"_out.txt")
        result_log_filename = os.path.join(dump_dir, dataset_technique_path, "logs", str(fold_no)+"_log.csv")
        result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", str(fold_no)+"_log.json")
        
        p_linear_pre_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "p_linear_pre_train", str(fold_no)+".pth")
        q_pre_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "q_pre_train", str(fold_no)+".pth")
        p_mlp_pre_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "p_mlp_pre_train", str(fold_no)+".pth")
        train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train", str(fold_no)+".pth") 
    
        logs = []
        
        if((pretrain == 'PQ') or (pretrain_strategy == "Q")):
            p_net_linear = Prediction_Net_Linear(input_dim, output_dim)
            s_net.p_net = Prediction_Net_Linear(input_dim, output_dim)
            
            p_net_linear.to(device)
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
    
    """
    
    