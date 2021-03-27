import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import Pdb

epsilon = 1e-6

class Prediction_Net_Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Prediction_Net_Linear, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform(self.fc1.weight)

    def forward(self, x):
        #Pdb().set_trace()
        #x = F.softmax(self.fc1(x))
        return self.fc1(x) 
        #return x
    
    def copy(self, net2):
        self.load_state_dict(net2.state_dict())

class Prediction_Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Prediction_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        #x = F.softmax(self.fc3(x))
        return x
    
    def copy(self, net2):
        self.load_state_dict(net2.state_dict())

class Phi_Net(nn.Module):
    def __init__(self, input_dim, output_dim, input_x):
        super(Phi_Net, self).__init__()
        self.input_x = input_x
        if(self.input_x):
            self.fc1 = nn.Linear(input_dim+output_dim, 512)
        else:
            self.fc1 = nn.Linear(input_dim, 512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)      

    def forward(self, x, p, targetSet, rl_technique):
        mask = targetSet>0
        
        p = p * targetSet
        
        if(self.input_x):
            x = torch.cat((x, p), 1)
        
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x[~mask] = -float('inf') + epsilon
        #if(rl_technique == "sample"):
        #    x = F.sigmoid(x)
        #else:
        #    x = F.softmax(x)
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
    
class G_Net(nn.Module):
    def __init__(self, x_dim, class_dim, method):
        super(G_Net, self).__init__()
        self.x_dim = x_dim
        self.class_dim = class_dim
        self.method = method
        if("loss_y" in self.method):
            self.fc1 = nn.Linear(class_dim, class_dim)
            for i in range(self.class_dim):
                self.fc1.weight[i][i].trainable = False
        elif('loss_xy' in self.method):
            self.fc1 = nn.Linear(self.x_dim + 20, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 20)
            self.fc4 = nn.Linear(20, self.class_dim)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(20)
            self.embedding = torch.nn.Embedding(self.class_dim, 20)
            torch.nn.init.xavier_uniform(self.fc1.weight)
            torch.nn.init.xavier_uniform(self.fc2.weight)
            torch.nn.init.xavier_uniform(self.fc3.weight)
            torch.nn.init.xavier_uniform(self.fc4.weight)
            
    
    def forward(self, inp):
        #inp is a (batchsize x class_dim) x class_dim Vector
        if("loss_y" in self.method):
            x = self.fc1(inp)
            return x
        elif('loss_xy' in self.method):
            #Pdb().set_trace()
            x = inp[0]
            y = inp[1].argmax(dim=1)
            y = self.embedding(y)
            x = torch.cat([x, y], dim=1)
            x = F.elu(self.bn1(self.fc1(x)))
            x = F.elu(self.bn2(self.fc2(x)))
            x = F.elu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            return x
            
    def setWeights(self, M):
        self.fc1.weight.data = M
    
    
class G_Net_Tie(nn.Module):
    def __init__(self, x_dim, class_dim, method):
        super(G_Net_Tie, self).__init__()
        self.x_dim = x_dim
        self.class_dim = class_dim
        self.method = method
        if("loss_y" in self.method):
            self.fc1 = nn.Linear(class_dim, class_dim)
            for i in range(self.class_dim):
                self.fc1.weight[i][i].trainable = False
        elif('loss_xy' in self.method):
            self.fc1 = nn.Linear(self.x_dim + 50, 128)
            self.fc2 = nn.Linear(128, 50)
            self.fc3 = nn.Linear(50, 50)
            #self.fc4 = nn.Linear(50, self.class_dim)
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(50)
            self.bn3 = nn.BatchNorm1d(50)
            self.embedding = torch.nn.Embedding(self.class_dim, 50)
            torch.nn.init.xavier_uniform(self.fc1.weight)
            torch.nn.init.xavier_uniform(self.fc2.weight)
            torch.nn.init.xavier_uniform(self.fc3.weight)
            
    
    def forward(self, inp):
        #inp is a (batchsize x class_dim) x class_dim Vector
        if("loss_y" in self.method):
            x = self.fc1(inp)
            return x
        elif('loss_xy' in self.method):
            #Pdb().set_trace()
            x = inp[0]
            y = inp[1].argmax(dim=1)
            y = self.embedding(y)
            x = torch.cat([x, y], dim=1)
            x = F.elu(self.bn1(self.fc1(x)))
            x = F.elu(self.bn2(self.fc2(x)))
            x = F.elu(self.bn3(self.fc3(x)))
            #print("Tying")
            x = torch.nn.functional.linear(x, self.embedding.weight)
            #x = self.fc4(x)
            return x
            
    def setWeights(self, M):
        self.fc1.weight.data = M
    
class G_Net_Full(nn.Module):
    def __init__(self, x_dim, class_dim, method):
        super(G_Net_Tie, self).__init__()
        self.x_dim = x_dim
        self.class_dim = class_dim
        self.method = method
        if("loss_y" in self.method):
            self.fc1 = nn.Linear(class_dim, class_dim)
            for i in range(self.class_dim):
                self.fc1.weight[i][i].trainable = False
        elif('loss_xy' in self.method):
            self.fc1 = nn.Linear(self.x_dim + 100, 128)
            self.fc2 = nn.Linear(128, 50)
            self.fc3 = nn.Linear(50, 50)
            self.fc4 = nn.Linear(50, 1)
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(50)
            self.bn3 = nn.BatchNorm1d(50)
            
            self.embedding = torch.nn.Embedding(self.class_dim, 50)
            torch.nn.init.xavier_uniform(self.fc1.weight)
            torch.nn.init.xavier_uniform(self.fc2.weight)
            torch.nn.init.xavier_uniform(self.fc3.weight)
            torch.nn.init.xavier_uniform(self.fc4.weight)
            
    
    def forward(self, inp):
        #inp is a (batchsize x class_dim) x class_dim Vector
        if("loss_y" in self.method):
            x = self.fc1(inp)
            return x
        elif('loss_xy' in self.method):
            x = inp[0]
            y = inp[1].argmax(dim=1)
            y = self.embedding(y)
            y_dash = inp[2].argmax(dim=1)
            y_dash = self.embedding(y_dash)
            x = torch.cat([x, y, y_dash], dim=1)
            x = F.elu(self.bn1(self.fc1(x)))
            x = F.elu(self.bn2(self.fc2(x)))
            x = F.elu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            return x
    
    
    
    
    
    
    
    
    
    
    
    
