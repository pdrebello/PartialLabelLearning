import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import scipy.io
from dataset import Dataset, loadTrain
from losses import  cc_loss, weighted_cc_loss, min_loss, naive_loss, iexplr_loss, regularized_cc_loss, sample_loss_function, sample_reward_function, select_loss_function, select_reward_function

from networks import Prediction_Net, Prediction_Net_Linear, Selection_Net, Phi_Net, G_Net, G_Net_Tie, G_Net_Full
import sys
from IPython.core.debugger import Pdb
import random
import csv
import os
import json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description = "Description for my parser")

parser.add_argument('--dataset', type=str, help="dataset")
parser.add_argument('--datasets', type=str, help="list of datasets")
parser.add_argument('--fold_no', type=int, help="fold number")
parser.add_argument('--dump_dir', type=str, help="dump directory for results")
parser.add_argument('--technique', type=str, help="training procedure")
parser.add_argument('--model', type=str, help="Use a 1 layer model for prediction?")
parser.add_argument('--pretrain_stategy', type=str, help="If RL, how to pretrain?")
parser.add_argument('--lambd', type=float, help="regularization cc_loss hyperparameter")

parser.add_argument('--pretrain_p', type=int, help="Pretrain P network")
parser.add_argument('--pretrain_q', type=int, help="Pretrain Q network")

parser.add_argument('--pretrain_p_perc', type=str, help="Pretrain P network percentage")
parser.add_argument('--shuffle', type=str, help="Experiment with datasets")
parser.add_argument('--optimizer', type=str, help="Optimizer: default Adam")
parser.add_argument('--batch_size', type=int, help="batch_size", default = 64)
parser.add_argument('--dataset_folder', type=str, help="dataset_folder")

parser.add_argument('--tie', default=0,type=int, help="tie embedding weights")
argument = parser.parse_args()
   

batch_size_train = argument.batch_size
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

epsilon = 1e-6

#Reproducibility
def set_random_seeds(random_seed):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

set_random_seeds(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epoch, train_loader, loss_function, p_net, p_optimizer, M = None):
    p_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        p_optimizer.zero_grad()
        #Pdb().set_trace() 
        output = p_net(data)
        
        if(M is not None):
            loss = loss_function(output, target, M)
        else:
            loss = loss_function(output, target)

        loss.backward()
        
        p_optimizer.step()
        
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def rl_train(epoch, train_loader, rl_technique, p_net, p_optimizer, s_net, s_optimizer):
    p_net.train()
    s_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        p_optimizer.zero_grad()
        s_optimizer.zero_grad()
        
        #prob output of prediction network
        p = p_net(data)
        q = s_net(data, target, rl_technique)
        
        if(rl_technique == 'exponential_rl'):
            
            with torch.no_grad():
                q_prob = torch.sigmoid(q)
                m = torch.distributions.bernoulli.Bernoulli(q_prob)
                a = m.sample()
            mask = target
            reward = sample_reward_function(p.detach(), q, a, mask)
            loss = sample_loss_function(p, q.detach(), a, mask)
            
            #old_loss = old_sample_loss_function(p, q.detach(), a, mask)
            #old_reward = old_sample_reward_function(p.detach(), q, a, mask)
            
            if((torch.isnan(loss)) or (torch.isinf(loss))):
                Pdb().set_trace()
            if((torch.isnan(reward)) or (torch.isinf(reward))):
                Pdb().set_trace()
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

def weighted_train_full(epoch, train_loader, p_net, p_optimizer, g_net, g_optimizer, method, class_dim):
    p_net.train()
    
    #class_dim = train_loader.__getitem__(0)[1].shape[1]
    row = np.asarray(list(range(class_dim)))
    #one_hot_gpu = torch.zeros((class_dim*class_dim, class_dim+class_dim))
    #one_hot_gpu = one_hot_gpu.to(device)
    
    y_gold = torch.arange(row.size)
    y_dash = torch.cat(class_dim*[y_gold])
    y_gold = y_gold.repeat_interleave(class_dim, dim=0)
    one_hot_gpu = torch.stack([y_gold, y_dash], dim=1)
    one_hot_gpu = one_hot_gpu.to(device)
    #one_hot_gpu[torch.arange(one_hot_gpu.shape[0]), y_gold] = 1
    #one_hot_gpu[torch.arange(one_hot_gpu.shape[0]), y_dash + class_dim] = 1
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        p_optimizer.zero_grad()
        output = p_net(data)
        
    
        batch = data.shape[0]
        
        
        one_hot = one_hot_gpu
        one_hot = one_hot.expand(batch, class_dim*class_dim, 2).reshape(batch*class_dim*class_dim, 2)
        #one_hot = one_hot.to(device)
        
        if("loss_xy" in method):
            oh = data.repeat_interleave(class_dim *class_dim, dim=0)
            one_hot = (oh, one_hot)
        g_output = g_net(one_hot)
        g_output = g_output.view(batch*class_dim, class_dim)
        log_sigmoid = nn.LogSigmoid()
        target_concat = target.repeat_interleave(class_dim, dim=0)
        #Pdb().set_trace()
        #g_output = log_sigmoid(g_output) * target_concat + (log_sigmoid(-g_output))*(1-target_concat)
        g_output = log_sigmoid(g_output) * target_concat
        g_output = g_output.sum(dim=1)
        
        split_g_output = g_output.view(batch, class_dim)
        
        if('iexplr' in method):
            #prob = torch.softmax(output, dim=1).detach()
            log_prob =  split_g_output+ torch.log_softmax(output, dim=1)
            prob = torch.exp(log_prob).detach()
            target_probs = (prob*target.float()).sum(dim=1)
            mask = ((target == 1) & (prob > epsilon))
            loss = -(prob[mask]*log_prob[mask]/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
            
        else:
        #cc_loss
            log_target_prob = split_g_output +  F.log_softmax(output, dim = 1)
            log_max_prob,max_prob_index = log_target_prob.max(dim=1)
            exp_argument = log_target_prob - log_max_prob.unsqueeze(dim=1)
            summ = (target*torch.exp(exp_argument)).sum(dim=1)
            log_total_prob = log_max_prob + torch.log(summ + epsilon)
            loss = (-1.0*log_total_prob).mean(dim=-1)
        
        
        
        loss.backward()
        
        p_optimizer.step()
        g_optimizer.step()
        
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def weighted_train(epoch, train_loader, p_net, p_optimizer, g_net, g_optimizer, method, class_dim):
    p_net.train()
    
    #class_dim = train_loader.__getitem__(0)[1].shape[1]
    row = np.asarray(list(range(class_dim)))
    one_hot_gpu = torch.zeros((row.size, class_dim))
    one_hot_gpu = one_hot_gpu.to(device)
    one_hot_gpu[torch.arange(row.size), row] = 1
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        """
        class_dim = target.shape[1]
        batch = data.shape[0]
        row = np.asarray(list(range(class_dim)))
        one_hot = np.zeros((row.size, class_dim))
        one_hot[torch.arange(row.size), row] = 1
        #one_hot = one_hot.expand(batch, class_dim, class_dim).reshape(batch*class_dim, class_dim)
        one_hot = np.broadcast_to(one_hot, (batch, class_dim, class_dim)).reshape(batch*class_dim, class_dim)
        
        if(method == 'weighted_loss_xy'):
            oh = data.numpy().repeat(class_dim, axis=0)
            one_hot = torch.cat([torch.from_numpy(oh).float(), torch.from_numpy(one_hot).float()], dim=1).float()
        else:
            one_hot = torch.from_numpy(one_hot).float()
        target_concat = torch.from_numpy(target.numpy().repeat(class_dim, axis=0)).float()
        target_concat = target_concat.to(device)
        data, target = data.to(device), target.to(device)
        p_optimizer.zero_grad()
        output = p_net(data)
        
        #Pdb().set_trace()
        """
        
        data, target = data.to(device), target.to(device)
        p_optimizer.zero_grad()
        output = p_net(data)
        
        #Pdb().set_trace()
        
        #For each training example, create class_dim repeats
        
        batch = data.shape[0]
        
        
        one_hot = one_hot_gpu
        one_hot = one_hot.expand(batch, class_dim, class_dim).reshape(batch*class_dim, class_dim)
        #one_hot = one_hot.to(device)
        
        if("loss_xy" in method):
            oh = data.repeat_interleave(class_dim, dim=0)
            one_hot = (oh, one_hot)
        g_output = g_net(one_hot)
        
        log_sigmoid = nn.LogSigmoid()
        target_concat = target.repeat_interleave(class_dim, dim=0)
        
        #g_output = log_sigmoid(g_output) * target_concat + (log_sigmoid(-g_output))*(1-target_concat)
        g_output = log_sigmoid(g_output) * target_concat
        g_output = g_output.sum(dim=1)
        
        split_g_output = g_output.view(batch, class_dim)
        
        if('iexplr' in method):
            #prob = torch.softmax(output, dim=1).detach()
            log_prob =  split_g_output+ torch.log_softmax(output, dim=1)
            prob = torch.exp(log_prob).detach()
            target_probs = (prob*target.float()).sum(dim=1)
            mask = ((target == 1) & (prob > epsilon))
            loss = -(prob[mask]*log_prob[mask]/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
            
        else:
        #cc_loss
            log_target_prob = split_g_output +  F.log_softmax(output, dim = 1)
            log_max_prob,max_prob_index = log_target_prob.max(dim=1)
            exp_argument = log_target_prob - log_max_prob.unsqueeze(dim=1)
            summ = (target*torch.exp(exp_argument)).sum(dim=1)
            log_total_prob = log_max_prob + torch.log(summ + epsilon)
            loss = (-1.0*log_total_prob).mean(dim=-1)
        
        
        
        loss.backward()
        
        p_optimizer.step()
        g_optimizer.step()
        
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

         
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

def q_accuracy_Subset(real_test_data, partial_test_data, q_net, rl_technique):
    q_net.eval()
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(zip(real_test_data, partial_test_data)):
            
            real_data = data[0][0]
            target = data[0][1]
            partial_data = data[1][0]
            partial_target = data[1][1]
            real_data, target, partial_data, partial_target = real_data.to(device), target.to(device), partial_data.to(device), partial_target.to(device) 
            assert(torch.equal(real_data, partial_data))
                
            output = q_net.forward(partial_data, partial_target, rl_technique)
            pred = output.data.max(1, keepdim=True)[1]
            correct += torch.gather(target, 1, pred).sum()
        #for data, target in test_data:
        #    data, target = data.to(device), target.to(device)
        #    
        #    pred = output.data.max(1, keepdim=True)[1]
        #    correct += torch.gather(target, 1, pred).sum()
    print((100. * float(correct) / len(real_test_data.dataset)))
    return (100. * float(correct) / len(real_test_data.dataset))
    #return 0

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


def getPretrainPEpochs(thr, logfile):
    thr = float(thr)/100.0
    dat = pd.read_json(logfile,orient = 'records',lines=True)
    best_val = dat['surrogate_val_acc'].iloc[-1]
    best_val
    dat = dat[dat['epoch'] >= 0]
    epsilon = 0.00001
    thr = thr - epsilon
    pretrain_till = dat['epoch'][dat['surrogate_val_acc']/best_val >= thr].iloc[0]
    return pretrain_till

def computeM(train_loader, output_dim, p_net):
    M = torch.zeros((output_dim,output_dim))
    M = M.to(device)
    
    den = torch.zeros(output_dim)
    den = den.to(device)
    p_net.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            predict = torch.softmax(p_net.forward(data), dim=1)
            idx = torch.argmax(predict, dim=-1)
            pred = torch.zeros( predict.shape )
            pred[np.arange(predict.shape[0]), idx] = 1
            
        pred = pred.to(device) 
        for i in range(output_dim):
            for j in range(output_dim):
                M[i][j] += (pred[:,i]*target[:,j]).sum()
        den += pred.sum(dim = 0)
    
    for i in range(output_dim):
        for j in range(output_dim):
            M[i][j] = (M[i][j]+epsilon)/(den[i]+output_dim*epsilon)
    M = torch.log(M/(1-M))
    return M
    
    
def main():
    
    dump_dir = argument.dump_dir
    filename = argument.dataset
    dataset_folder = argument.dataset_folder
    datasets = argument.datasets
    datasets = [str(item) for item in datasets.split(',')]
    fold_no = argument.fold_no
    
    technique = argument.technique
    
    model = argument.model
    if(model is None):
        model = '3layer'
    
    pretrain_p = True if argument.pretrain_p == 1 else False
    pretrain_q = True if argument.pretrain_q == 1 else False
    pretrain_p_perc = argument.pretrain_p_perc
    
    k = 10
    
    pretrain_p_epochs = 3
    pretrain_q_epochs = 50
    
    shuffle_name = argument.shuffle
    
        
    loss_techniques = ["fully_supervised", "cc_loss", "min_loss", "naive_loss", "iexplr_loss", 'regularized_cc_loss']
    
    if((argument.optimizer is None) or (argument.optimizer == "Adam")):
        optimizer = lambda x: torch.optim.Adam(x,weight_decay = 0.000001)
        #optimizer = torch.optim.Adam
    elif(argument.optimizer == 'SGD'):
        optimizer = lambda x: torch.optim.SGD(x, lr=0.1, momentum=0.9)
        #optimizer = torch.optim.SGD
    else:
        optimizer = torch.optim.Adam
        
    
    for filename in datasets:
        if(filename in ['lost', 'MSRCv2']):
            n_epochs = 1000
        elif(filename in ['BirdSong']):
            n_epochs = 1500
        else:
            n_epochs = 150
        #n_epochs = 2
        if(shuffle_name is not None):
            filename = filename +"_"+shuffle_name
        train_dataset, real_train_dataset, val_dataset, real_val_dataset, test_dataset, real_test_dataset, input_dim, output_dim = loadTrain(os.path.join(dataset_folder,filename+".mat"), fold_no, k)
        
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
        
        logs = []
        
        if(technique in loss_techniques):
            dataset_technique_path = os.path.join(filename, model, technique, str(fold_no))
            if(technique == "cc_loss"):
                loss_function = cc_loss
            elif(technique == "min_loss"):
                loss_function = min_loss
            elif(technique == "naive_loss"):
                loss_function = naive_loss
            elif(technique == "iexplr_loss"):
                loss_function = iexplr_loss
            elif(technique == "regularized_cc_loss"):
                lambd = argument.lambd
                loss_function = lambda x, y : regularized_cc_loss(lambd, x, y)
                dataset_technique_path = os.path.join(filename, model, technique+"_"+str(lambd), str(fold_no))
            elif(technique == "fully_supervised"):
                #loss_function = ce_loss
                loss_function = naive_loss
                #loss_function = min_loss
                train_loader = real_train_loader
                test_loader = real_test_loader
                val_loader = real_val_loader
                
            set_random_seeds(1) 
            if(model == "1layer"):
                p_net = Prediction_Net_Linear(input_dim, output_dim)
            else:
                p_net = Prediction_Net(input_dim, output_dim)
                
            p_net.to(device)
            p_optimizer = optimizer(p_net.parameters())
            
            
            
            result_filename = os.path.join(dump_dir, dataset_technique_path, "results", "out.txt")
            result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", "log.json")
            train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best.pth") 
            
            best_val = 0
            best_val_epoch = -1
            for epoch in range(1,n_epochs+1):
                train(epoch, train_loader, loss_function, p_net, p_optimizer)
                surrogate_train_acc = p_accuracy(train_loader, p_net)
                real_train_acc = p_accuracy(real_train_loader, p_net)
                surrogate_val_acc = p_accuracy(val_loader, p_net)
                real_val_acc = p_accuracy(real_val_loader, p_net)
                print(real_val_acc)
                
                
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
            
            
            log = {'epoch':-1, 'best_epoch': best_val_epoch, 'phase': 'test', 
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
        elif("weighted" in technique ):
            dataset_technique_path = os.path.join(filename, model, technique, str(fold_no))
            
            p_net = Prediction_Net(input_dim, output_dim)
            p_net.to(device)
            p_optimizer = optimizer(p_net.parameters())
            dataset_pretrain_technique_path = os.path.join(filename, model, "cc_loss", str(fold_no))
            train_checkpoint = os.path.join(dump_dir, dataset_pretrain_technique_path, "models", "train_best.pth") 
            checkpoint = torch.load(train_checkpoint)
            p_net.load_state_dict(checkpoint['p_net_state_dict'])
            
            if("full" in technique):
                g_net = G_Net_Full(input_dim, output_dim, technique)
            elif("tie" in technique):
                print("Here")
                g_net = G_Net_Tie(input_dim, output_dim, technique)
            else:
                g_net = G_Net(input_dim, output_dim, technique)
            if("loss_y"  in technique):
                M = computeM(train_loader, output_dim, p_net) 
                M = M.to(device)
                g_net.setWeights(M)
            g_net.to(device)
            g_optimizer = optimizer(g_net.parameters())
            
            result_filename = os.path.join(dump_dir, dataset_technique_path, "results", "out.txt")
            result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", "log.json")
            train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best.pth") 
            
            best_val = 0
            best_val_epoch = -1
            for epoch in range(1,n_epochs+1):
                if('full' in technique):
                    weighted_train_full(epoch, train_loader, p_net, p_optimizer, g_net, g_optimizer, technique, output_dim)
                else:
                    weighted_train(epoch, train_loader, p_net, p_optimizer, g_net, g_optimizer, technique, output_dim)
                
                surrogate_train_acc = p_accuracy(train_loader, p_net)
                real_train_acc = p_accuracy(real_train_loader, p_net)
                surrogate_val_acc = p_accuracy(val_loader, p_net)
                real_val_acc = p_accuracy(real_val_loader, p_net)
                
                print(surrogate_train_acc)
                print(real_val_acc)
                
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
            
            
            log = {'epoch':-1, 'best_epoch': best_val_epoch, 'phase': 'test', 
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
                    
                    
                    
        elif((technique == "linear_rl") or (technique == "exponential_rl")):    
            loss_function = cc_loss
            
            if(model == "1layer"):
                p_net = Prediction_Net_Linear(input_dim, output_dim)
            else:
                p_net = Prediction_Net(input_dim, output_dim)   
            p_net.to(device)
            p_optimizer = optimizer(p_net.parameters())
            s_net = Selection_Net(input_dim, output_dim, True)
            s_net.to(device)
            
            overall_strategy = technique
            if(pretrain_p):
                overall_strategy += "_P"
                if(pretrain_p_perc is not None):
                    overall_strategy += pretrain_p_perc
            if(pretrain_q):
                overall_strategy += "_Q"
            print(overall_strategy)
            dataset_technique_path = os.path.join(filename, model, overall_strategy, str(fold_no))
            
            result_filename = os.path.join(dump_dir, dataset_technique_path, "results", "out.txt")
            result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", "log.json")
            
            logs = []
            
            #Pretraining of P Network
            if(pretrain_p):
                if(pretrain_p_perc == "best"):
                    dataset_technique_path_load = os.path.join(filename, model, "cc_loss", str(fold_no))
                    best_checkpoint = os.path.join(dump_dir, dataset_technique_path_load, "models", "train_best.pth")
                    checkpoint = torch.load(best_checkpoint)
                    p_net.load_state_dict(checkpoint['p_net_state_dict'])
                
                else:
                    dataset_technique_path_load = os.path.join(filename, model, "cc_loss", str(fold_no))
                    pretrain_logfile = os.path.join(dump_dir, dataset_technique_path_load, "logs", "log.json")
                    pretrain_p_epochs = getPretrainPEpochs(pretrain_p_perc, pretrain_logfile)
                    train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "pretrain_p.pth")
                    for epoch in range(1,pretrain_p_epochs+1):
                        train(epoch, train_loader, loss_function, p_net, p_optimizer)
                        surrogate_train_acc = p_accuracy(train_loader, p_net)
                        real_train_acc = p_accuracy(real_train_loader, p_net)
                        surrogate_val_acc = p_accuracy(val_loader, p_net)
                        real_val_acc = p_accuracy(real_val_loader, p_net)
                        
                        log = {'epoch':epoch, 'best_epoch': None,'phase': 'pretrain_p', 
                                   'surrogate_train_acc': surrogate_train_acc, 'real_train_acc': real_train_acc, 
                                   'surrogate_val_acc': surrogate_val_acc, 'real_val_acc': real_val_acc, 
                                   'surrogate_test_acc': None, 'real_test_acc': None, 
                                   'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                                   'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                                   'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                                   'info': dataset_technique_path}
                        logs.append(log)
                    save_checkpoint(pretrain_p_epochs, surrogate_val_acc, p_net, p_optimizer, None, None, train_checkpoint)
            
            #Pretraining of Q Network
            if(pretrain_q):
                p_net_linear = Prediction_Net_Linear(input_dim, output_dim)
                s_net.p_net = Prediction_Net_Linear(input_dim, output_dim)
                
                p_optimizer_linear = optimizer(p_net_linear.parameters())
                s_optimizer = optimizer(s_net.parameters())
                
                p_net_linear.to(device)
                s_net.to(device)   
                
                train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "pretrain_p_linear.pth") 
                
                dataset_technique_path_load = os.path.join(filename, "1layer", "cc_loss", str(fold_no))
                pretrain_logfile = os.path.join(dump_dir, dataset_technique_path_load, "logs", "log.json")
                pretrain_p_epochs = getPretrainPEpochs(pretrain_p_perc, pretrain_logfile)
                
                for epoch in range(1,pretrain_p_epochs+1):
                    train(epoch, train_loader, loss_function, p_net_linear, p_optimizer_linear)
                    surrogate_train_acc = p_accuracy(train_loader, p_net_linear)
                    real_train_acc = p_accuracy(real_train_loader, p_net_linear)
                    surrogate_val_acc = p_accuracy(val_loader, p_net_linear)
                    real_val_acc = p_accuracy(real_val_loader, p_net_linear)
                    
                    log = {'epoch':epoch, 'best_epoch': None,'phase': 'pretrain_p_linear', 
                               'surrogate_train_acc': surrogate_train_acc, 'real_train_acc': real_train_acc, 
                               'surrogate_val_acc': surrogate_val_acc, 'real_val_acc': real_val_acc, 
                               'surrogate_test_acc': None, 'real_test_acc': None, 
                               'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                               'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                               'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                               'info': dataset_technique_path}
                    logs.append(log)
                save_checkpoint(pretrain_p_epochs, surrogate_val_acc, p_net_linear, p_optimizer_linear, None, None, train_checkpoint)
        
                
                for param in s_net.p_net.parameters():
                    param.requires_grad = False
                
                train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "pretrain_q.pth") 
                for epoch in range(1,pretrain_q_epochs+1):
                    rl_train(epoch, train_loader, technique, p_net_linear, p_optimizer_linear, s_net, s_optimizer)
                    surrogate_train_acc = p_accuracy(train_loader, p_net_linear)
                    real_train_acc = p_accuracy(real_train_loader, p_net_linear)
                    surrogate_val_acc = p_accuracy(val_loader, p_net_linear)
                    real_val_acc = p_accuracy(real_val_loader, p_net_linear)
                    
                    q_surrogate_train_acc = q_accuracy(train_loader, s_net, technique)
                    q_real_train_acc = q_accuracy(real_train_loader, s_net, technique)
                    q_surrogate_val_acc = q_accuracy(val_loader, s_net, technique)
                    q_real_val_acc = q_accuracy(real_val_loader, s_net, technique)
                    
                    log = {'epoch':epoch, 'best_epoch': None,'phase': 'pretrain_q', 
                               'surrogate_train_acc': surrogate_train_acc, 'real_train_acc': real_train_acc, 
                               'surrogate_val_acc': surrogate_val_acc, 'real_val_acc': real_val_acc, 
                               'surrogate_test_acc': None, 'real_test_acc': None, 
                               'q_surrogate_train_acc': q_surrogate_train_acc, 'q_real_train_acc': q_real_train_acc, 
                               'q_surrogate_val_acc': q_surrogate_val_acc, 'q_real_val_acc': q_real_val_acc, 
                               'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                               'info': dataset_technique_path}
                    logs.append(log)
                save_checkpoint(pretrain_q_epochs, surrogate_val_acc, p_net_linear, p_optimizer_linear, s_net, s_optimizer, train_checkpoint)
             
                
            #JOINT RL TRAINING
            if(model == "1layer"):
                s_net.p_net = Prediction_Net_Linear(input_dim, output_dim)
            else:
                s_net.p_net = Prediction_Net(input_dim, output_dim) 
                
            s_optimizer = optimizer(s_net.parameters())
            s_net.to(device)  
            
            for param in s_net.p_net.parameters():
                param.requires_grad = False
            
            train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best.pth") 
            best_val = 0
            best_val_epoch = -1
            
            for epoch in range(1,n_epochs+1):
                rl_train(epoch, train_loader, technique, p_net, p_optimizer, s_net, s_optimizer)
                surrogate_train_acc = p_accuracy(train_loader, p_net)
                real_train_acc = p_accuracy(real_train_loader, p_net)
                surrogate_val_acc = p_accuracy(val_loader, p_net)
                real_val_acc = p_accuracy(real_val_loader, p_net)
                
                q_surrogate_train_acc = q_accuracy(train_loader, s_net, technique)
                q_real_train_acc = q_accuracy(real_train_loader, s_net, technique)
                q_surrogate_val_acc = q_accuracy(val_loader, s_net, technique)
                q_real_val_acc = q_accuracy(real_val_loader, s_net, technique)
                
                log = {'epoch':epoch, 'best_epoch': best_val_epoch,'phase': 'train', 
                           'surrogate_train_acc': surrogate_train_acc, 'real_train_acc': real_train_acc, 
                           'surrogate_val_acc': surrogate_val_acc, 'real_val_acc': real_val_acc, 
                           'surrogate_test_acc': None, 'real_test_acc': None, 
                           'q_surrogate_train_acc': q_surrogate_train_acc, 'q_real_train_acc': q_real_train_acc, 
                           'q_surrogate_val_acc': q_surrogate_val_acc, 'q_real_val_acc': q_real_val_acc, 
                           'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                           'info': dataset_technique_path}
                logs.append(log)
                
                
                if(surrogate_val_acc > best_val):
                    best_val = surrogate_val_acc
                    best_val_epoch = epoch
                    save_checkpoint(epoch, surrogate_val_acc, p_net, p_optimizer, s_net, s_optimizer, train_checkpoint)
             
            checkpoint = torch.load(train_checkpoint)
            p_net.load_state_dict(checkpoint['p_net_state_dict'])
            s_net.phi_net.load_state_dict(checkpoint['s_net_state_dict'])
            s_net.p_net.copy(p_net)
            
            surrogate_train_acc = p_accuracy(train_loader, p_net)
            real_train_acc = p_accuracy(real_train_loader, p_net)
            surrogate_val_acc = p_accuracy(val_loader, p_net)
            real_val_acc = p_accuracy(real_val_loader, p_net)
            surrogate_test_acc = p_accuracy(test_loader, p_net)
            real_test_acc = p_accuracy(real_test_loader, p_net)
            
            q_surrogate_train_acc = q_accuracy(train_loader, s_net, technique)
            q_real_train_acc = q_accuracy(real_train_loader, s_net, technique)
            q_surrogate_val_acc = q_accuracy(val_loader, s_net, technique)
            q_real_val_acc = q_accuracy(real_val_loader, s_net, technique)
            q_surrogate_test_acc = q_accuracy(test_loader, s_net, technique)
            q_real_test_acc = q_accuracy(real_test_loader, s_net, technique)
            
            log = {'epoch':-1, 'best_epoch': best_val_epoch, 'phase': 'test', 
                           'surrogate_train_acc': surrogate_train_acc, 'real_train_acc': real_train_acc, 
                           'surrogate_val_acc': surrogate_val_acc, 'real_val_acc': real_val_acc, 
                           'surrogate_test_acc': surrogate_test_acc, 'real_test_acc': real_test_acc, 
                           'q_surrogate_train_acc': q_surrogate_train_acc, 'q_real_train_acc': q_real_train_acc, 
                           'q_surrogate_val_acc': q_surrogate_val_acc, 'q_real_val_acc': q_real_val_acc, 
                           'q_surrogate_test_acc': q_surrogate_test_acc, 'q_real_test_acc': q_real_test_acc, 
                           'info': dataset_technique_path}
            logs.append(log)
        
            os.makedirs(os.path.dirname(result_log_filename_json), exist_ok=True)
            with open(result_log_filename_json, "w") as file:
                for log in logs:
                    json.dump(log, file)
                    file.write("\n")
        
if __name__ == "__main__":
    main()   
    
