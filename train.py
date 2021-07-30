import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import scipy.io
from dataset import Dataset, loadTrain
from losses import  cc_loss, weighted_cc_loss, min_loss, naive_loss, iexplr_loss, regularized_cc_loss, sample_loss_function, sample_reward_function, select_loss_function, select_reward_function, svm_loss, cour_loss
from networks import LSTM
from networks import Prediction_Net,LeNet5, Prediction_Net_Linear, Selection_Net, Phi_Net, G_Net_Tie, G_Net_Full, G_Net_Hyperparameter, G_Net_Y, G_Net_XY
import sys
from IPython.core.debugger import Pdb
import random
import csv
import os
import json
import argparse
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR

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
parser.add_argument('--pretrain_g', type=int, help="Pretrain G network")

parser.add_argument('--pretrain', type=int, default = 0, help="Pretrain Weighted network")

parser.add_argument('--lr', type=float, help="learning rate", default = 0.001)
parser.add_argument('--weight_decay', type=float, help="weight decay", default = 0.000001)

parser.add_argument('--neg_sample', type=int, default = 0, help="negative sampling while pretraining g_net")

parser.add_argument('--pretrain_p_perc', type=int, help="Pretrain P network percentage")
parser.add_argument('--shuffle', type=str, help="Experiment with datasets")
parser.add_argument('--optimizer', type=str, help="Optimizer: default Adam")
parser.add_argument('--batch_size', type=int, help="batch_size", default = 64)
parser.add_argument('--dataset_folder', type=str, help="dataset_folder")

parser.add_argument('--val_metric',default ='acc',type=str, help="validation accuracy metric: loss or accuracy")

parser.add_argument('--freeze50',default =0,type=int, help="freeze p_net for first 50 epochs of joint training")


parser.add_argument('--tie', default=0,type=int, help="tie embedding weights")
argument = parser.parse_args()
   

batch_size_train = argument.batch_size
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

epsilon = 10e-12

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
        #Pdb().set_trace()
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

def weighted_train(epoch, train_loader, p_net, p_optimizer, g_net, g_optimizer, method, class_dim, freeze_epochs=50):
    freeze50 = (argument.freeze50 == 1) and (epoch <= freeze_epochs)
    
    #print(p_net.fc1.weight[0])
    
    if(freeze50):
        p_net.eval()
    else:
        p_net.train()
        
    row = np.asarray(list(range(class_dim)))
    one_hot_gpu = torch.zeros((row.size, class_dim))
    one_hot_gpu = one_hot_gpu.to(device)
    one_hot_gpu[torch.arange(row.size), row] = 1
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        
        data, target = data.to(device), target.to(device)
        p_optimizer.zero_grad()
        
        if(freeze50):
            with torch.no_grad():
                output = p_net(data)
        else:
            output = p_net(data)
        batch = data.shape[0]
        
        one_hot = one_hot_gpu
        one_hot = one_hot.expand(batch, class_dim, class_dim).reshape(batch*class_dim, class_dim)
        #Pdb().set_trace()
        
        #class_dim is EOS
        #class_dim +1 is NULL
        
        if("loss_xy_lstm" in method):
            
            max_req = int(torch.max(target.sum(dim=1)).item())
            relevant_indices = target.nonzero()
            
            input_x = data[relevant_indices[:,0]]
            input_y = relevant_indices[:,1]
            input_y_onehot = torch.zeros((input_y.shape[0], class_dim+2))
            input_y_onehot = input_y_onehot.to(device)
            
            input_y = torch.unsqueeze(input_y, dim = -1)
            input_y_onehot.scatter_(1, input_y, 1)
            input_y = input_y_onehot
            
            
            input_x = torch.unsqueeze(input_x, dim = 0)
            input_x = torch.repeat_interleave(input_x, max_req+1, dim=0)
            
            input_y = torch.unsqueeze(input_y, dim = 0)
            input_y = torch.repeat_interleave(input_y, max_req+1, dim=0)
            
            target_set = target[relevant_indices[:,0]]
            
            target_set2 = torch.ones(target_set.shape[0], max_req)
            target_set2 = target_set2.to(device)
            target_set = torch.cat([target_set, target_set2], dim=1)
            
            idx = torch.arange(target_set.shape[1], 0, -1).to(device).float()
            tmp2 = target_set * idx
            indices = torch.topk(tmp2, k = max_req+1, dim=1)[1]
            #mask = (indices > output_dim)
            indices[indices > class_dim] = class_dim+1
            indices = torch.transpose(indices, dim0=0, dim1=1)
            indices = torch.unsqueeze(indices, dim = -1)
            
            target_set = torch.zeros(indices.shape[0], indices.shape[1], class_dim+2)
            target_set = target_set.to(device)
            target_set = target_set.to(device)
            target_set = target_set.scatter(-1, indices, 1)
            
            #Pdb().set_trace()
            g_output = torch.zeros(input_x.shape[1], 1)
            g_output = g_output.to(device)
            
            g_net.hidden_cell = (torch.zeros(1, input_x.shape[1], g_net.hidden_layer_size).to(device),
                                torch.zeros(1, input_x.shape[1] , g_net.hidden_layer_size).to(device))
            
            
            #Pdb().set_trace()
            for seq_step in range(max_req+1):
                x = input_x[seq_step]
                y = input_y[seq_step]
                s = target_set[seq_step]
                
                
                    
                g_optimizer.zero_grad()
                
                
                if(seq_step == 0):
                    y_pred = g_net(x.clone(), y.clone(), y.clone())
                else:
                    y_pred = g_net(x.clone(), y.clone(), target_set[seq_step-1].clone())
                
                
                #index = s.argmax(dim=1).unsqueeze(-1)
                y_pred = y_pred.log_softmax(dim=1)
                
                
                loss = nn.CrossEntropyLoss(reduce = False, ignore_index = class_dim+1)
                
                g_output -= (loss(y_pred, s.argmax(dim=1))).unsqueeze(-1)
                #g_output += (torch.gather(y_pred, 1, index))
                #if(epoch == 200):
                #    Pdb().set_trace()
                
            g_output = g_output.flatten()
            
            temp = torch.zeros((batch*class_dim))
            temp = temp.to(device)
            project_index = relevant_indices[:,0] * class_dim + relevant_indices[:,1]
            temp[project_index] = g_output
            g_output = temp
            
            
        elif("loss_xy" in method):
            
            relevant_indices = target.nonzero()
            input_x = data[relevant_indices[:,0]]
            input_y = relevant_indices[:,1]
            
            g_output = g_net((input_x,input_y),device)
            log_sigmoid = nn.LogSigmoid()
            
            target_concat = target[relevant_indices[:,0]]
            g_output = log_sigmoid(g_output) * target_concat + (log_sigmoid(-g_output))*(1-target_concat)
            g_output = g_output.sum(dim=1)
            
            temp = torch.zeros((batch*class_dim))
            temp = temp.to(device)
            project_index = relevant_indices[:,0] * class_dim + relevant_indices[:,1]
            temp[project_index] = g_output
            g_output = temp
        else:    
            g_output = g_net(one_hot)
            
            log_sigmoid = nn.LogSigmoid()
            target_concat = target.repeat_interleave(class_dim, dim=0)
            g_output = log_sigmoid(g_output) * target_concat + (log_sigmoid(-g_output))*(1-target_concat)
            
            g_output = g_output.sum(dim=1)
        
        split_g_output = g_output.view(batch, class_dim)
        
        if('iexplr' in method):
            #log_prob =  split_g_output + torch.log_softmax(output, dim=1)
            #prob = torch.exp(log_prob).detach()
            #prob = log_prob.detach()
            #target_probs = (prob*target.float()).sum(dim=1)
            #mask = ((target == 1) & (abs(prob) > epsilon))
            #Pdb().set_trace()
            #loss = -(prob[mask]*log_prob[mask]/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
            logp = torch.log_softmax(output, dim=1)
            logg = split_g_output
            
            mask = (target == 1).float()
            #Pdb().set_trace()
            
            weights = torch.exp(logp+logg).detach() * mask
            norm = weights.sum(dim=1).unsqueeze(1).expand_as(mask)
            weights = weights/norm
            
            p_loss = -(weights * logp).sum(dim=1).mean()
            g_loss = -(weights * logg).sum(dim=1).mean()
            loss = p_loss
            
            if(not(freeze50)):
                p_loss.backward()
            
            g_loss.backward()  
            
            if(batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tG Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), p_loss.item(), g_loss.item()))
        else:
            
            mask = (target != 1)
            
            log_target_prob2 = split_g_output
            log_target_prob_for_max2 = (split_g_output).masked_fill(mask,-float('inf'))
            log_max_prob2,max_prob_index2 = log_target_prob_for_max2.max(dim=1)
            exp_argument2 = log_target_prob2 - log_max_prob2.unsqueeze(dim=1)
            
            norm = log_max_prob2 +torch.log((target*torch.exp(exp_argument2*target)).sum(dim=1))
            #print(norm)
            #norm = torch.ones_like(norm)
            #norm = norm * np.log(pow(0.2,23))
            #log_target_prob = split_g_output  +  F.log_softmax(output, dim = 1)
            #log_target_prob_for_max = (split_g_output  +  F.log_softmax(output, dim = 1)).masked_fill(mask,-float('inf'))
            norm = norm.detach() 
            log_target_prob = split_g_output  - norm.unsqueeze(dim=1)  +  F.log_softmax(output, dim = 1)
            log_target_prob_for_max = (split_g_output - norm.unsqueeze(dim=1) +  F.log_softmax(output, dim = 1)).masked_fill(mask,-float('inf'))
            
            #log_target_prob = split_g_output  +  F.log_softmax(output, dim = 1)
            #log_target_prob_for_max = (split_g_output +  F.log_softmax(output, dim = 1)).masked_fill(mask,-float('inf'))
            
            log_max_prob,max_prob_index = log_target_prob_for_max.max(dim=1)
            exp_argument = log_target_prob - log_max_prob.unsqueeze(dim=1) 
            #exp_argument = exp_argument
            
            #print(torch.exp(norm))
            
            summ = (target*torch.exp(exp_argument*target)).sum(dim=1)
            #summ = summ/norm
            log_total_prob = log_max_prob + torch.log(summ + epsilon)
            loss = (-1.0*log_total_prob).mean(dim=-1)
            #if(torch.isnan(loss)):
            #    Pdb().set_trace()
            loss.backward()
            
            if(batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        
        p_optimizer.step()
        if(not(freeze50)):
            p_optimizer.step()
        g_optimizer.step()
        
        

         
def p_accuracy(test_data, p_net, loss_function=cc_loss):
    p_net.eval()
    correct = 0
    loss = 0
    confidence = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = p_net.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            
            prob = torch.softmax(output, dim=1)
            confidence += torch.gather(prob, 1, pred).sum()
            
            correct += torch.gather(target, 1, pred).sum()
            this_loss =  (loss_function(output, target))
            loss += this_loss * len(data)
         
    return {'acc':(100. * float(correct.item()) / len(test_data.dataset)), 'loss':loss.item()/(len(test_data.dataset)), 'confidence': confidence.item()/len(test_data.dataset)}

def p_accuracy_weighted(test_data, p_net, g_net, method, class_dim):
    p_net.eval()
    row = np.asarray(list(range(class_dim)))
    one_hot_gpu = torch.zeros((row.size, class_dim))
    one_hot_gpu = one_hot_gpu.to(device)
    one_hot_gpu[torch.arange(row.size), row] = 1
    
    correct = 0
    loss = 0
    confidence = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data):
            data, target = data.to(device), target.to(device)
            output = p_net(data)
            pred = output.data.max(1, keepdim=True)[1]
            
            prob = torch.softmax(output, dim=1)
            confidence += torch.gather(prob, 1, pred).sum()
            
            correct += torch.gather(target, 1, pred).sum()
            
            batch = data.shape[0]
            
            one_hot = one_hot_gpu
            one_hot = one_hot.expand(batch, class_dim, class_dim).reshape(batch*class_dim, class_dim)
            if("loss_xy_lstm" in method):
            
                max_req = int(torch.max(target.sum(dim=1)).item())
                relevant_indices = target.nonzero()
                
                input_x = data[relevant_indices[:,0]]
                input_y = relevant_indices[:,1]
                input_y_onehot = torch.zeros((input_y.shape[0], class_dim+2))
                input_y_onehot = input_y_onehot.to(device)
                
                input_y = torch.unsqueeze(input_y, dim = -1)
                input_y_onehot.scatter_(1, input_y, 1)
                input_y = input_y_onehot
                
                
                input_x = torch.unsqueeze(input_x, dim = 0)
                input_x = torch.repeat_interleave(input_x, max_req+1, dim=0)
                
                input_y = torch.unsqueeze(input_y, dim = 0)
                input_y = torch.repeat_interleave(input_y, max_req+1, dim=0)
                
                target_set = target[relevant_indices[:,0]]
                
                target_set2 = torch.ones(target_set.shape[0], max_req)
                target_set2 = target_set2.to(device)
                target_set = torch.cat([target_set, target_set2], dim=1)
                
                idx = torch.arange(target_set.shape[1], 0, -1).to(device).float()
                tmp2 = target_set * idx
                indices = torch.topk(tmp2, k = max_req+1, dim=1)[1]
                #mask = (indices > output_dim)
                indices[indices > class_dim] = class_dim+1
                indices = torch.transpose(indices, dim0=0, dim1=1)
                indices = torch.unsqueeze(indices, dim = -1)
                
                target_set = torch.zeros(indices.shape[0], indices.shape[1], class_dim+2)
                target_set = target_set.to(device)
                target_set = target_set.to(device)
                target_set = target_set.scatter(-1, indices, 1)
                
                #Pdb().set_trace()
                g_output = torch.zeros(input_x.shape[1], 1)
                g_output = g_output.to(device)
                
                g_net.hidden_cell = (torch.zeros(1, input_x.shape[1], g_net.hidden_layer_size).to(device),
                                    torch.zeros(1, input_x.shape[1] , g_net.hidden_layer_size).to(device))
                
                
                #Pdb().set_trace()
                for seq_step in range(max_req+1):
                    x = input_x[seq_step]
                    y = input_y[seq_step]
                    s = target_set[seq_step]
                    
                    
                    if(seq_step == 0):
                        y_pred = g_net(x.clone(), y.clone(), y.clone())
                    else:
                        y_pred = g_net(x.clone(), y.clone(), target_set[seq_step-1].clone())
                    
                    
                    #index = s.argmax(dim=1).unsqueeze(-1)
                    y_pred = y_pred.log_softmax(dim=1)
                    
                    
                    loss = nn.CrossEntropyLoss(reduce = False, ignore_index = class_dim+1)
                    
                    g_output -= (loss(y_pred, s.argmax(dim=1))).unsqueeze(-1)
                    
                g_output = g_output.flatten()
                
                temp = torch.zeros((batch*class_dim))
                temp = temp.to(device)
                project_index = relevant_indices[:,0] * class_dim + relevant_indices[:,1]
                temp[project_index] = g_output
                g_output = temp
                
            elif("loss_xy" in method):
                relevant_indices = target.nonzero()
                input_x = data[relevant_indices[:,0]]
                input_y = relevant_indices[:,1]
                
                g_output = g_net((input_x,input_y), device)
                log_sigmoid = nn.LogSigmoid()
                
                target_concat = target[relevant_indices[:,0]]
                
                
                g_output = log_sigmoid(g_output) * target_concat + (log_sigmoid(-g_output))*(1-target_concat)
                g_output = g_output.sum(dim=1)
                
                temp = torch.zeros((batch*class_dim))
                temp = temp.to(device)
                project_index = relevant_indices[:,0] * class_dim + relevant_indices[:,1]
                temp[project_index] = g_output
                g_output = temp
                
                #zer = torch.zeros_like(batch, g_output.shape[1])
                #zer.index_add_(0, relevant_indices[:,0], g_output)
                #oh = data.repeat_interleave(class_dim, dim=0)
                #one_hot = (oh, one_hot)
                
                
                
                
            else:    
                g_output = g_net(one_hot)
                
                log_sigmoid = nn.LogSigmoid()
                target_concat = target.repeat_interleave(class_dim, dim=0)
                #g_output = log_sigmoid(g_output) * target_concat
                #g_output = torch.log_softmax(g_output, dim=1) * target_concat
                g_output = log_sigmoid(g_output) * target_concat + (log_sigmoid(-g_output))*(1-target_concat)
                
                g_output = g_output.sum(dim=1)
            
            split_g_output = g_output.view(batch, class_dim)
            
            
            if('iexplr' in method):
                logp = torch.log_softmax(output, dim=1)
                logg = split_g_output
                
                mask = (target == 1).float()
                #Pdb().set_trace()
                
                weights = torch.exp(logp+logg).detach() * mask
                norm = weights.sum(dim=1).unsqueeze(1).expand_as(mask)
                weights = weights/norm
                
                p_loss = -(weights * logp).sum(dim=1).mean()
                #g_loss = -(weights * logg).sum(dim=1).mean()
                loss = p_loss
                #g_loss = -(weights * logg).sum(dim=1).mean()
            else:
                mask = (target != 1)
            
                log_target_prob2 = split_g_output
                log_target_prob_for_max2 = (split_g_output).masked_fill(mask,-float('inf'))
                log_max_prob2,max_prob_index2 = log_target_prob_for_max2.max(dim=1)
                exp_argument2 = log_target_prob2 - log_max_prob2.unsqueeze(dim=1)
                
                
                norm = log_max_prob2 +torch.log((target*torch.exp(exp_argument2*target)).sum(dim=1))
                #print(norm)
                
                log_target_prob = split_g_output  - norm.unsqueeze(dim=1)  +  F.log_softmax(output, dim = 1)
                log_target_prob_for_max = (split_g_output - norm.unsqueeze(dim=1) +  F.log_softmax(output, dim = 1)).masked_fill(mask,-float('inf'))
                log_max_prob,max_prob_index = log_target_prob_for_max.max(dim=1)
                exp_argument = log_target_prob - log_max_prob.unsqueeze(dim=1) 
                #exp_argument = exp_argument
                
                #print(torch.exp(norm))
                
                summ = (target*torch.exp(exp_argument*target)).sum(dim=1)
                #summ = summ/norm
                log_total_prob = log_max_prob + torch.log(summ + epsilon)
                loss = (-1.0*log_total_prob).mean(dim=-1)
    return {'acc':(100. * float(correct.item()) / len(test_data.dataset)), 'loss':loss.item()/(len(test_data.dataset)), 'confidence': confidence.item()/len(test_data.dataset)}

def save_checkpoint(epoch, val_acc, p_net, p_optimizer, s_net, s_optimizer, filename, g_net = None, g_optimizer = None):
    if(s_net is None):
        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'p_net_state_dict': p_net.state_dict(),
            'p_optimizer': p_optimizer.state_dict(),
            's_net_state_dict': None,
            's_optimizer': None,
            'g_optimizer': g_optimizer,
            'g_net':g_net
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'p_net_state_dict': p_net.state_dict(),
            'p_optimizer': p_optimizer.state_dict(),
            's_net_state_dict': s_net.phi_net.state_dict(),
            's_optimizer': s_optimizer.state_dict(),
            'g_optimizer': g_optimizer,
            'g_net':g_net
        }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)


def getPretrainPEpochs(thr, logfile):
    thr = float(thr)/100.0
    print(logfile)
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


def pretrainG(epochs, train_loader, output_dim, p_net, g_net, g_optimizer, class_dim, neg_sample):
    
    for epoch in range(1, epochs+1):
        count = 0
        p_net.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                predict = torch.softmax(p_net.forward(data), dim=1)
                idx = torch.argmax(predict, dim=-1)
                pred = torch.zeros( predict.shape )
                pred[np.arange(predict.shape[0]), idx] = 1
                #Pdb().set_trace()
                mask = predict > -1
                conf = predict > 0.95
                #Pdb().set_trace()
               # neg_conf = (predict < 0.5) * conf.float().sum(dim=1).unsqueeze(1).expand_as(mask)
                
                #conf = predict.gather(-1, idx.long().view(-1,1))
                #confidence_mask = conf > 0.95
                #count += confidence_mask.long().sum()
            #Pdb().set_trace()
            pred = pred.to(device) 
            
            max_req = int(torch.max(target.sum(dim=1)).item())
            relevant_indices = target.nonzero()
            
            input_x = data[relevant_indices[:,0]]
            input_y = relevant_indices[:,1]
            input_y_onehot = torch.zeros((input_y.shape[0], class_dim+2))
            input_y_onehot = input_y_onehot.to(device)
            
            input_y = torch.unsqueeze(input_y, dim = -1)
            input_y_onehot.scatter_(1, input_y, 1)
            input_y = input_y_onehot
            
            
            input_x = torch.unsqueeze(input_x, dim = 0)
            input_x = torch.repeat_interleave(input_x, max_req+1, dim=0)
            
            input_y = torch.unsqueeze(input_y, dim = 0)
            input_y = torch.repeat_interleave(input_y, max_req+1, dim=0)
            
            target_set = target[relevant_indices[:,0]]
            
            target_set2 = torch.ones(target_set.shape[0], max_req)
            target_set2 = target_set2.to(device)
            target_set = torch.cat([target_set, target_set2], dim=1)
            
            idx = torch.arange(target_set.shape[1], 0, -1).to(device).float()
            tmp2 = target_set * idx
            indices = torch.topk(tmp2, k = max_req+1, dim=1)[1]
            #mask = (indices > output_dim)
            indices[indices > class_dim] = class_dim+1
            indices = torch.transpose(indices, dim0=0, dim1=1)
            indices = torch.unsqueeze(indices, dim = -1)
            
            target_set = torch.zeros(indices.shape[0], indices.shape[1], class_dim+2)
            target_set = target_set.to(device)
            target_set = target_set.to(device)
            target_set = target_set.scatter(-1, indices, 1)
            
            #Pdb().set_trace()
            g_output = torch.zeros(input_x.shape[1], 1)
            g_output = g_output.to(device)
            
            g_net.hidden_cell = (torch.zeros(1, input_x.shape[1], g_net.hidden_layer_size).to(device),
                                torch.zeros(1, input_x.shape[1] , g_net.hidden_layer_size).to(device))
            
            
            for seq_step in range(max_req+1):
                x = input_x[seq_step]
                y = input_y[seq_step]
                s = target_set[seq_step]
                
                
                    
                g_optimizer.zero_grad()
                
                
                if(seq_step == 0):
                    y_pred = g_net(x.clone(), y.clone(), y.clone())
                else:
                    y_pred = g_net(x.clone(), y.clone(), target_set[seq_step-1].clone())
                
                
                #index = s.argmax(dim=1).unsqueeze(-1)
                y_pred = y_pred.log_softmax(dim=1)
                
                
                loss = nn.CrossEntropyLoss(reduce = False, ignore_index = class_dim+1)
                
                g_output -= (loss(y_pred, s.argmax(dim=1))).unsqueeze(-1)
                #g_output += (torch.gather(y_pred, 1, index))
                #if(epoch == 200):
                #    Pdb().set_trace()
                
            g_output = g_output.flatten()
            
            temp = torch.zeros((batch*class_dim))
            temp = temp.to(device)
            project_index = relevant_indices[:,0] * class_dim + relevant_indices[:,1]
            temp[project_index] = g_output
            g_output = temp
            
            split_g_output = g_output.view(batch, class_dim)
            
            logg = split_g_output
            
            mask = (target == 1).float()
            #Pdb().set_trace()
            
            weights = torch.exp(logg).detach() * mask
            norm = weights.sum(dim=1).unsqueeze(1).expand_as(mask)
            weights = weights/norm
            
            #Pdb().set_trace()
            g_loss = -(weights * logg * conf.float())
            #if(torch.isnan(g_loss.sum())):
            #    Pdb().set_trace()
            if(neg_sample):
                sample = (weights * logg * neg_conf.float())
                if(torch.isnan(sample.sum())):
                    Pdb().set_trace()
                if(torch.isinf(sample.sum())):
                    Pdb().set_trace()
                g_loss += (weights * logg * neg_conf.float())
                
            g_loss = g_loss.sum(dim=1).mean()
            g_loss.backward()  
            
            g_optimizer.step()
            #print('Pretrain {}: Loss {}'.format(epoch, g_loss))
        print("G Loss: {}".format(g_loss.item()))
        
    #return M
def lr_lambda(epoch: int):
    #if(100 < epoch < 1000):
    #    return 0.1
    if(200 > epoch):
        return pow(0.98, epoch-1)
    else:
        return (0.01)

# Optimizer has lr set to 0.01
    
def main():
    
    dump_dir = argument.dump_dir
    filename = argument.dataset
    dataset_folder = argument.dataset_folder
    datasets = argument.datasets
    datasets = [str(item) for item in datasets.split(',')]
    fold_no = argument.fold_no
    val_metric = argument.val_metric
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
    
        
    loss_techniques = ["fully_supervised", "cc_loss", "min_loss", "naive_loss", "iexplr_loss", 'regularized_cc_loss','cour_loss', 'svm_loss']
    
    if((argument.optimizer is None) or (argument.optimizer == "Adam")):
        wd = argument.weight_decay
        lr = argument.lr
        optimizer = lambda x: torch.optim.Adam(x,weight_decay = wd, lr = lr)
        #optimizer = torch.optim.Adam
    elif(argument.optimizer == 'SGD'):
        lr = argument.lr
        wd = argument.weight_decay
        optimizer = lambda x: torch.optim.SGD(x, lr=lr, momentum=0.9, weight_decay = wd)
    else:
        optimizer = torch.optim.Adam
        
    
  
    
    for filename in datasets:
        freeze_epochs = 50
        if('MSRCv2' in filename):
            n_epochs = 1000
        elif('lost' in filename):
            n_epochs = 1000
        elif(('BirdSong' in filename)):
            n_epochs = 600
            
        else:
            n_epochs = 200
            freeze_epochs = 20
        #n_epochs = 50
        print(n_epochs)
        #freeze_epochs = 1
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
        #Pdb().set_trace()
        logs = []
        flag = False
        for i in loss_techniques:
            if((i in technique) and (not("weighted" in technique))):
                flag = True
        if(flag):
            dataset_technique_path = os.path.join(filename, model, technique, str(fold_no))
            if("cc_loss" in technique):
                loss_function = cc_loss
            elif("min_loss" in technique):
                loss_function = min_loss
            elif("naive_loss" in technique):
                loss_function = naive_loss
            elif("cour_loss" in technique):
                loss_function = cour_loss
            elif("svm_loss" in technique):
                loss_function = svm_loss
            elif("iexplr_loss" in technique):
                loss_function = iexplr_loss
            elif("regularized_cc_loss" in technique):
                lambd = argument.lambd
                loss_function = lambda x, y : regularized_cc_loss(lambd, x, y)
                dataset_technique_path = os.path.join(filename, model, technique+"_"+str(lambd), str(fold_no))
            elif("fully_supervised" in technique):
                loss_function = naive_loss
                train_loader = real_train_loader
                test_loader = real_test_loader
                val_loader = real_val_loader
                
            set_random_seeds(1) 
            if(model == "1layer"):
                p_net = Prediction_Net_Linear(input_dim, output_dim)
            elif(model == "LeNet"):
                p_net = LeNet5(input_dim, output_dim)
            else:
                p_net = Prediction_Net(input_dim, output_dim)
                
            p_net.to(device)
            p_optimizer = optimizer(p_net.parameters())
            p_scheduler = LambdaLR(p_optimizer, lr_lambda=lr_lambda)   
            
            
            result_filename = os.path.join(dump_dir, dataset_technique_path, "results", "out.txt")
            result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", "log.json")
            train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best.pth") 
            train_real_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best_real.pth") 
            
            if(val_metric == 'loss'):
                best_val = np.inf
                best_real_val = np.inf
            else:
                best_val = 0
                best_real_val = 0
            
            best_val_epoch = -1
            best_real_val_epoch = -1
            
            best_val_epoch = -1
            for epoch in range(1,n_epochs+1):
                train(epoch, train_loader, loss_function, p_net, p_optimizer)
                
                surrogate_train = p_accuracy(train_loader, p_net, loss_function)
                real_train = p_accuracy(real_train_loader, p_net, loss_function)
                surrogate_val = p_accuracy(val_loader, p_net, loss_function)
                real_val = p_accuracy(real_val_loader, p_net, loss_function)
                p_scheduler.step()
                for param_group in p_optimizer.param_groups:
                    print(param_group['lr'])
                
                
                log = {'epoch':epoch, 'best_epoch': best_val_epoch,'phase': 'train', 
                           'surrogate_train_acc': surrogate_train['acc'], 'real_train_acc': real_train['acc'], 
                           'surrogate_val_acc': surrogate_val['acc'], 'real_val_acc': real_val['acc'], 
                           'surrogate_test_acc': None, 'real_test_acc': None, 
                           'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                           'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                           'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                           'info': dataset_technique_path,
                           'surrogate_train_loss': surrogate_train['loss'], 'real_train_loss': real_train['loss'], 
                           'surrogate_val_loss': surrogate_val['loss'], 'real_val_loss': real_val['loss'], 
                           'surrogate_test_loss': None, 'real_test_loss': None,
                           'train_confidence':surrogate_train['confidence'],
                           'val_confidence':surrogate_val['confidence'],
                           'test_confidence':None}
                logs.append(log)
                
                current_val = surrogate_val[val_metric]
                current_real_val = real_val[val_metric]
                print(current_val)
                #print(real_test['acc'])
                if(((val_metric == 'acc') and (current_val > best_val)) or ((val_metric == 'loss') and (current_val < best_val))):
                    best_val = current_val
                    best_val_epoch = epoch
                    save_checkpoint(epoch, current_val, p_net, p_optimizer, None, None, train_checkpoint)
                
                if(((val_metric == 'acc') and (current_real_val > best_real_val)) or ((val_metric == 'loss') and (current_real_val < best_real_val))):
                    best_real_val = current_real_val
                    best_real_val_epoch = epoch
                    save_checkpoint(epoch, current_real_val, p_net, p_optimizer, None, None, train_real_checkpoint)
                    
                epoch_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_{}.pth".format(epoch)) 
                if('Soccer' in filename):
                    if((epoch <= 20) or (epoch%20 == 0)):
                        save_checkpoint(epoch, current_val, p_net, p_optimizer, None, None, epoch_train_checkpoint)
            
                else:
                    if((epoch <= 20) or (epoch%50 == 0)):
                        save_checkpoint(epoch, current_val, p_net, p_optimizer, None, None, epoch_train_checkpoint)
            
            
            checkpoint = torch.load(train_checkpoint)
            p_net.load_state_dict(checkpoint['p_net_state_dict'])
            surrogate_train = p_accuracy(train_loader, p_net, loss_function)
            real_train = p_accuracy(real_train_loader, p_net, loss_function)
            surrogate_val = p_accuracy(val_loader, p_net, loss_function)
            real_val = p_accuracy(real_val_loader, p_net, loss_function)
            surrogate_test = p_accuracy(test_loader, p_net, loss_function)
            real_test = p_accuracy(real_test_loader, p_net, loss_function)
            
            
            
            
            log = {'epoch':-1, 'best_epoch': best_val_epoch, 'phase': 'test', 
                           'surrogate_train_acc': surrogate_train['acc'], 'real_train_acc': real_train['acc'], 
                           'surrogate_val_acc': surrogate_val['acc'], 'real_val_acc': real_val['acc'], 
                           'surrogate_test_acc': surrogate_test['acc'], 'real_test_acc': real_test['acc'], 
                           'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                           'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                           'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                           'info': dataset_technique_path,
                           'surrogate_train_loss': surrogate_train['loss'], 'real_train_loss': real_train['loss'], 
                           'surrogate_val_loss': surrogate_val['loss'], 'real_val_loss': real_val['loss'], 
                           'surrogate_test_loss': surrogate_test['loss'], 'real_test_loss': real_test['loss'],
                           'train_confidence':surrogate_train['confidence'],
                           'val_confidence':surrogate_val['confidence'],
                           'test_confidence':surrogate_test['confidence']}
            logs.append(log)
            
            
            
            checkpoint = torch.load(train_real_checkpoint)
            p_net.load_state_dict(checkpoint['p_net_state_dict'])
            
            surrogate_train = p_accuracy(train_loader, p_net, loss_function)
            real_train = p_accuracy(real_train_loader, p_net, loss_function)
            surrogate_val = p_accuracy(val_loader, p_net, loss_function)
            real_val = p_accuracy(real_val_loader, p_net, loss_function)
            surrogate_test = p_accuracy(test_loader, p_net, loss_function)
            real_test = p_accuracy(real_test_loader, p_net, loss_function)
            
            
            log = {'epoch':-2, 'best_epoch': best_real_val_epoch, 'phase': 'test', 
                           'surrogate_train_acc': surrogate_train['acc'], 'real_train_acc': real_train['acc'], 
                           'surrogate_val_acc': surrogate_val['acc'], 'real_val_acc': real_val['acc'], 
                           'surrogate_test_acc': surrogate_test['acc'], 'real_test_acc': real_test['acc'], 
                           'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                           'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                           'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                           'info': dataset_technique_path,
                           'surrogate_train_loss': surrogate_train['loss'], 'real_train_loss': real_train['loss'], 
                           'surrogate_val_loss': surrogate_val['loss'], 'real_val_loss': real_val['loss'], 
                           'surrogate_test_loss': surrogate_test['loss'], 'real_test_loss': real_test['loss'],
                           'train_confidence':surrogate_train['confidence'],
                           'val_confidence':surrogate_val['confidence'],
                           'test_confidence':surrogate_test['confidence']}
            logs.append(log)
            
            
            
            os.makedirs(os.path.dirname(result_log_filename_json), exist_ok=True)
            with open(result_log_filename_json, "w") as file:
                for log in logs:
                    json.dump(log, file)
                    file.write("\n")
        elif("weighted" in technique ):
            if("fully_supervised" in technique):
                train_loader = real_train_loader
                test_loader = real_test_loader
                val_loader = real_val_loader
                
            dataset_technique_path = os.path.join(filename, model, technique, str(fold_no))
            
            p_net = Prediction_Net(input_dim, output_dim)
            p_net.to(device)
            p_optimizer = optimizer(p_net.parameters())
            p_scheduler = LambdaLR(p_optimizer, lr_lambda=lr_lambda)   
            
            pretrain_p_perc = argument.pretrain_p_perc
            print(pretrain_p_perc)
            print(type(pretrain_p_perc))
            if((argument.pretrain_p == 1) and (pretrain_p_perc > 0)):
                print("ENTERED")
                if('fully_supervised' in technique):
                    dataset_pretrain_technique_path = os.path.join(filename, model, "fully_supervised_{}_{}_{}".format(argument.optimizer,argument.lr,argument.weight_decay), str(fold_no))
                else:
                    dataset_pretrain_technique_path = os.path.join(filename, model, "cc_loss_{}_{}_{}".format(argument.optimizer,argument.lr,argument.weight_decay), str(fold_no))
                    if(argument.lr == 1):
                        dataset_pretrain_technique_path = os.path.join(filename, model, "cc_loss_{}_{}_{}".format(argument.optimizer,int(argument.lr),argument.weight_decay), str(fold_no))
                
                if(pretrain_p_perc == 100):
                    train_checkpoint = os.path.join(dump_dir, dataset_pretrain_technique_path, "models", "train_best.pth") 
                    
                else:
                    print("desired")
                    logfile = os.path.join(dump_dir, dataset_pretrain_technique_path, "logs", "log.json") 
                    desired_thr = getPretrainPEpochs(pretrain_p_perc, logfile)
                    
                    train_checkpoint = os.path.join(dump_dir, dataset_pretrain_technique_path, "models", "train_"+str(desired_thr)+".pth") 
                    while(not(os.path.exists(train_checkpoint))):
                        desired_thr -= 1
                        if(desired_thr <= 1):
                            desired_thr = 1
                            break
                     #   print(train_checkpoint)
                    train_checkpoint = os.path.join(dump_dir, dataset_pretrain_technique_path, "models", "train_"+str(desired_thr)+".pth") 
                    print(desired_thr)
                    print("LOADING")
                
                
                checkpoint = torch.load(train_checkpoint)
                p_net.load_state_dict(checkpoint['p_net_state_dict'])
                
            
            
            
            if(("full" in technique) and not ("fully" in technique)):
                g_net = G_Net_Full(input_dim, output_dim, technique)
            elif("tie" in technique):
                g_net = G_Net_Tie(input_dim, output_dim, technique)
            elif("hyperparameter" in technique):
                g_net = G_Net_Hyperparameter(input_dim, output_dim, technique)
            
            elif("_xy_lstm" in technique):
                g_net = LSTM(input_dim, output_dim, technique)
            elif("_xy" in technique):
                g_net = G_Net_XY(input_dim, output_dim, technique)    
            else:
                g_net = G_Net_Y(input_dim, output_dim, technique)
            
            
            #if("loss_y"  in technique):
            #    M = computeM(train_loader, output_dim, p_net) 
            #    M = M.to(device)
            #    g_net.setWeights(M)
            g_net.to(device)
            #Pdb().set_trace()
            g_optimizer = optimizer(g_net.parameters())
            
            
            g_scheduler = LambdaLR(g_optimizer, lr_lambda=lr_lambda) 
            
            
            if(argument.pretrain_g == 1):
                pretrainG(freeze_epochs, train_loader, output_dim, p_net, g_net, g_optimizer, output_dim, argument.neg_sample)
            
            result_filename = os.path.join(dump_dir, dataset_technique_path, "results", "out.txt")
            result_log_filename_json = os.path.join(dump_dir, dataset_technique_path, "logs", "log.json")
            train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best.pth") 
            train_real_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_best_real.pth") 
            
            if(val_metric == 'loss'):
                best_val = np.inf
                best_real_val = np.inf
            else:
                best_val = 0
                best_real_val = 0
            
            best_val_epoch = -1
            best_real_val_epoch = -1
            
            
            for epoch in range(1,n_epochs+1):
                weighted_train(epoch, train_loader, p_net, p_optimizer, g_net, g_optimizer, technique, output_dim, freeze_epochs)
                surrogate_train = p_accuracy_weighted(train_loader, p_net, g_net, technique, output_dim)
                real_train = p_accuracy_weighted(real_train_loader, p_net, g_net, technique, output_dim)
                surrogate_val = p_accuracy_weighted(val_loader, p_net, g_net, technique, output_dim)
                real_val = p_accuracy_weighted(real_val_loader, p_net, g_net, technique, output_dim)
                
                g_scheduler.step()
                p_scheduler.step()
                
                
                log = {'epoch':epoch, 'best_epoch': best_val_epoch,'phase': 'train', 
                           'surrogate_train_acc': surrogate_train['acc'], 'real_train_acc': real_train['acc'], 
                           'surrogate_val_acc': surrogate_val['acc'], 'real_val_acc': real_val['acc'], 
                           'surrogate_test_acc': None, 'real_test_acc': None, 
                           'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                           'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                           'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                           'info': dataset_technique_path,
                           'surrogate_train_loss': surrogate_train['loss'], 'real_train_loss': real_train['loss'], 
                           'surrogate_val_loss': surrogate_val['loss'], 'real_val_loss': real_val['loss'], 
                           'surrogate_test_loss': None, 'real_test_loss': None,
                           'train_confidence':surrogate_train['confidence'],
                           'val_confidence':surrogate_val['confidence'],
                           'test_confidence':None}
                logs.append(log)
                current_val = surrogate_val[val_metric]
                current_real_val = real_val[val_metric]
                print(current_val)
                #print(real_test['acc'])
                if(((val_metric == 'acc') and (current_val > best_val)) or ((val_metric == 'loss') and (current_val < best_val))):
                    best_val = current_val
                    best_val_epoch = epoch
                    save_checkpoint(epoch, current_val, p_net, p_optimizer, None, None, train_checkpoint, g_net = g_net, g_optimizer = g_optimizer)
                    
                if(((val_metric == 'acc') and (current_real_val > best_real_val)) or ((val_metric == 'loss') and (current_real_val < best_real_val))):
                    best_real_val = current_real_val
                    best_real_val_epoch = epoch
                    save_checkpoint(epoch, current_val, p_net, p_optimizer, None, None, train_real_checkpoint, g_net = g_net, g_optimizer = g_optimizer)
                
                epoch_train_checkpoint = os.path.join(dump_dir, dataset_technique_path, "models", "train_{}.pth".format(epoch)) 
                if('Soccer' in filename):
                    if((epoch <= 20) or (epoch%20 == 0)):
                        save_checkpoint(epoch, current_val, p_net, p_optimizer, None, None, epoch_train_checkpoint, g_net = g_net, g_optimizer = g_optimizer)
            
                else:
                    if((epoch <= 20) or (epoch%50 == 0)):
                        save_checkpoint(epoch, current_val, p_net, p_optimizer, None, None, epoch_train_checkpoint, g_net = g_net, g_optimizer = g_optimizer)
            
            checkpoint = torch.load(train_checkpoint)
            p_net.load_state_dict(checkpoint['p_net_state_dict'])
            
            surrogate_train = p_accuracy_weighted(train_loader, p_net, g_net, technique, output_dim)
            real_train = p_accuracy_weighted(real_train_loader, p_net, g_net, technique, output_dim)
            surrogate_val = p_accuracy_weighted(val_loader, p_net, g_net, technique, output_dim)
            real_val = p_accuracy_weighted(real_val_loader, p_net, g_net, technique, output_dim)
            surrogate_test = p_accuracy_weighted(test_loader, p_net, g_net, technique, output_dim)
            real_test = p_accuracy_weighted(real_test_loader, p_net, g_net, technique, output_dim)
            
            
            log = {'epoch':-1, 'best_epoch': best_val_epoch, 'phase': 'test', 
                           'surrogate_train_acc': surrogate_train['acc'], 'real_train_acc': real_train['acc'], 
                           'surrogate_val_acc': surrogate_val['acc'], 'real_val_acc': real_val['acc'], 
                           'surrogate_test_acc': surrogate_test['acc'], 'real_test_acc': real_test['acc'], 
                           'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                           'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                           'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                           'info': dataset_technique_path,
                           'surrogate_train_loss': surrogate_train['loss'], 'real_train_loss': real_train['loss'], 
                           'surrogate_val_loss': surrogate_val['loss'], 'real_val_loss': real_val['loss'], 
                           'surrogate_test_loss': surrogate_test['loss'], 'real_test_loss': real_test['loss'],
                           'train_confidence':surrogate_train['confidence'],
                           'val_confidence':surrogate_val['confidence'],
                           'test_confidence':surrogate_test['confidence']}
            logs.append(log)
            
            
            checkpoint = torch.load(train_real_checkpoint)
            p_net.load_state_dict(checkpoint['p_net_state_dict'])
            
            surrogate_train = p_accuracy_weighted(train_loader, p_net, g_net, technique, output_dim)
            real_train = p_accuracy_weighted(real_train_loader, p_net, g_net, technique, output_dim)
            surrogate_val = p_accuracy_weighted(val_loader, p_net, g_net, technique, output_dim)
            real_val = p_accuracy_weighted(real_val_loader, p_net, g_net, technique, output_dim)
            surrogate_test = p_accuracy_weighted(test_loader, p_net, g_net, technique, output_dim)
            real_test = p_accuracy_weighted(real_test_loader, p_net, g_net, technique, output_dim)
            
            
            log = {'epoch':-2, 'best_epoch': best_real_val_epoch, 'phase': 'test', 
                           'surrogate_train_acc': surrogate_train['acc'], 'real_train_acc': real_train['acc'], 
                           'surrogate_val_acc': surrogate_val['acc'], 'real_val_acc': real_val['acc'], 
                           'surrogate_test_acc': surrogate_test['acc'], 'real_test_acc': real_test['acc'], 
                           'q_surrogate_train_acc': None, 'q_real_train_acc': None, 
                           'q_surrogate_val_acc': None, 'q_real_val_acc': None, 
                           'q_surrogate_test_acc': None, 'q_real_test_acc': None, 
                           'info': dataset_technique_path,
                           'surrogate_train_loss': surrogate_train['loss'], 'real_train_loss': real_train['loss'], 
                           'surrogate_val_loss': surrogate_val['loss'], 'real_val_loss': real_val['loss'], 
                           'surrogate_test_loss': surrogate_test['loss'], 'real_test_loss': real_test['loss'],
                           'train_confidence':surrogate_train['confidence'],
                           'val_confidence':surrogate_val['confidence'],
                           'test_confidence':surrogate_test['confidence']}
            logs.append(log)
            
            os.makedirs(os.path.dirname(result_log_filename_json), exist_ok=True)
            with open(result_log_filename_json, "w") as file:
                for log in logs:
                    json.dump(log, file)
                    file.write("\n")

        elif(("linear_rl" in technique) or ("exponential_rl"in technique)):    
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
                    overall_strategy += str(pretrain_p_perc)
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
                
                
                current_val = surrogate_val_acc[val_metric]
                
                if(((val_metric == 'acc') and (current_val > best_val)) or ((val_metric == 'loss') and (current_val < best_val))):
                    best_val = current_val
                    best_val_epoch = epoch
                    save_checkpoint(epoch, current_val, p_net, p_optimizer, s_net, s_optimizer, train_checkpoint)
             
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
    
