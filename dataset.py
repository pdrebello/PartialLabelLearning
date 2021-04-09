import scipy.io
import torch
import numpy as np
import random
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy import genfromtxt
from IPython.core.debugger import Pdb
import scipy.stats as stats


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels.astype(np.float32)
        self.data = data.astype(np.float32)
        
  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        X = np.asarray(self.data[index]).flatten()
        
        y = np.asarray(self.labels[index]).flatten()
       

        return X, y
    
class DatasetAnalysis(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, partials, target):
        'Initialization'
        self.data = data.astype(np.float32)
        self.partials = partials.astype(np.float32)
        self.target = target.astype(np.float32)

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        X = np.asarray(self.data[index]).flatten()
        p = np.asarray(self.partials[index]).flatten()
        y = np.asarray(self.target[index]).flatten()
       

        return X, p, y

class ConvDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels.astype(np.float32)
        self.data = data.astype(np.float32)

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        X = np.asarray(self.data[index])
        
        y = np.asarray(self.labels[index]).flatten()
       

        return X, y

   
def prepTrain(filename):  
    mat = scipy.io.loadmat("datasets/"+filename)
    if(filename == ('Soccer Player.mat')):
        target = mat["target"].T
        partials = mat["partial_target"].T
    else:
        target = mat["target"].todense().T
        partials = mat["partial_target"].todense().T
    
    data = mat["data"]
    
    
    combine = np.concatenate([data, partials, target], axis=1)
    
    np.random.shuffle(combine)
    dat = combine[:, :data.shape[1]]
    partials = combine[:, data.shape[1]:data.shape[1]+partials.shape[1]]
    target = combine[:, data.shape[1]+partials.shape[1]:data.shape[1]+partials.shape[1]+target.shape[1]]
    
    
    
    
    with open("datasets/"+filename+".pkl", "wb") as f:
        pickle.dump(dat, f)
        pickle.dump(partials, f)
        pickle.dump(target, f)
    #tr = list(dat)
    #for count,i in enumerate(tr):
     #   for j in range(count+1, len(tr)):
    #        if((tr[count] == tr[j]).all()):
    #            print(j)
    #train_dataset = Dataset(data[0:9*tenth], partials[0:9*tenth])
    #test_dataset = Dataset(data[9*tenth:], target[9*tenth:])
    #real_train_dataset = Dataset(data[0:9*tenth], target[0:9*tenth])

    #return train_dataset, test_dataset, real_train_dataset, data.shape[1], partials.shape[1]
def remake(filename, newname):
    with open("datasets/"+filename+".mat.pkl", "rb") as f:
        data = pickle.load(f)
        partials = pickle.load(f)
        target = pickle.load(f)
        
    new_target = np.zeros_like(target)
    
    #print(type(partials))
    for i in range(partials.shape[0]):
        indices = []
        for j in range(partials.shape[1]):
            
            if(partials[i,j] == 1):
                indices.append(j)
        rand = random.choice(indices)
        new_target[i,rand] = 1
        
    with open("datasets/"+filename+newname+".mat.pkl", "wb") as f:
        pickle.dump(data, f)
        pickle.dump(partials, f)
        pickle.dump(new_target, f)
        
        
def remakeCC(filename, add_number):
    with open("datasets/"+filename+".mat.pkl", "rb") as f:
        data = pickle.load(f)
        partials = pickle.load(f)
        target = pickle.load(f)
        
    new_partials = np.copy(partials)
    print(new_partials.shape[1])
    choices = range(new_partials.shape[1])
    
    for i in range(partials.shape[0]):
        fill_up_indices = random.sample(choices, add_number)
        
        for j in fill_up_indices:
            new_partials[i,j] = 1
            #print(new_partials[i,j])   
    with open("datasets/"+filename+"_"+str(add_number)+".mat.pkl", "wb") as f:
        pickle.dump(data, f)
        pickle.dump(new_partials, f)
        pickle.dump(target, f)
        
def loadTrain(filename, fold_no, k):  
    
    with open(filename+".pkl", "rb") as f:
        data = pickle.load(f)
        partials = pickle.load(f)
        target = pickle.load(f)
    if(not('MNIST' in filename)):
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
    split = int(data.shape[0]/k)
    
    train_data_list = []
    train_target_list = []
    train_partials_list = []
    
    test_data_list = []
    test_target_list = []
    test_partials_list = []
    
    val_data_list = []
    val_target_list = []
    val_partials_list = []
    
    if('MNIST' in filename):
        test_data_list.append(data[60000 : 70000])
        test_target_list.append(target[60000 : 70000])
        test_partials_list.append(partials[60000 : 70000])
        
        val_data_list.append(data[54000: 60000])
        val_target_list.append(target[54000: 60000])
        val_partials_list.append(partials[54000: 60000])
        
        train_data_list.append(data[0 :54000])
        train_target_list.append(target[0 :54000])
        train_partials_list.append(partials[0 :54000])
    else:
        for i in range(k):
            if(fold_no == i):
                test_data_list.append(data[i*split : (i+1)*split])
                test_target_list.append(target[i*split : (i+1)*split])
                test_partials_list.append(partials[i*split : (i+1)*split])
            elif(i == (fold_no+1)%k):
                val_data_list.append(data[i*split : (i+1)*split])
                val_target_list.append(target[i*split : (i+1)*split])
                val_partials_list.append(partials[i*split : (i+1)*split])
            else:
                train_data_list.append(data[i*split : (i+1)*split])
                train_target_list.append(target[i*split : (i+1)*split])
                train_partials_list.append(partials[i*split : (i+1)*split])
    
    train_data = np.vstack(train_data_list)
    train_target = np.vstack(train_target_list)
    train_partials = np.vstack(train_partials_list)
    
    test_data = np.vstack(test_data_list)
    test_target = np.vstack(test_target_list)
    test_partials = np.vstack(test_partials_list)
    
    val_data = np.vstack(val_data_list)
    val_target = np.vstack(val_target_list)
    val_partials = np.vstack(val_partials_list)
    
    train_dataset = Dataset(train_data, train_partials)
    test_dataset = Dataset(test_data, test_partials)
    val_dataset = Dataset(val_data, val_partials)
    real_train_dataset = Dataset(train_data, train_target)
    real_test_dataset = Dataset(test_data, test_target)
    real_val_dataset = Dataset(val_data, val_target)
    
    #return train_data, test_data
    return train_dataset, real_train_dataset, val_dataset, real_val_dataset, test_dataset, real_test_dataset, data.shape[1], partials.shape[1]

def makeTransition(filename):
    """matrix = [[3,7,1,23,43,12,9,34,4,2],
              [6,4,34,9,12,4,56,4,2,1],
              [30,6,4,34,2,56,3,1,8,2],
              [89,34,2,14,24,1,75,89,2,8],
              [4,4,3,2,4,5,1,2,5,4],
              [65,34,12,4,76,3,1,23,9,1],
              [56,1,23,5,34,11,7,34,2,10],
              [7,90,23,80,32,9,4,1,32,56],
              [90,54,23,12,5,24,24,90,2,22],
              [1,2,3,5,2,90,4,2,23,43]]"""
    with open("datasets/"+filename+".mat.pkl", "rb") as f:
        data = pickle.load(f)
        partials = pickle.load(f)
    labels = partials.shape[1]
    lower, upper = 0, 1
    mu, sigma = 0.5, 0.25
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    matrix = np.zeros((labels, labels))
    for i in range(labels):
        for j in range(labels):
            x = X.rvs(1)
            if(x <=0.5):
                x = 0.5-x
            else:
                x = 1-x +0.45
            if('Soccer' in filename):
                if(x>0.94):
                    matrix[i][j] = x/3
                else:
                    matrix[i][j] = x/50
            else:
                if(x>0.8):
                    matrix[i][j] = x/2
                else:
                    matrix[i][j] = x/10
    print(matrix.max(axis=1))       
    for i in range(labels):
        #m = max(matrix[i])
        #m = m+np.random.randint(10,30)
        #for j in range(10):
        #    matrix[i][j] = matrix[i][j]/float(m)
        matrix[i][i] = 1
    transition = matrix
    with open("datasets/"+"transition2_"+filename+".pkl", "wb") as f:
        pickle.dump(transition, f)

def flip(p):
    return(random.random() < p)
    
def makeMNIST():
    train_data = genfromtxt('datasets/mnist_train.csv', delimiter=',')
    test_data = genfromtxt('datasets/mnist_test.csv', delimiter=',')
    
    X_train = train_data[1:,1:]
    Y_train = train_data[1:,0].astype(int)
    X_test = test_data[1:, 1:]
    Y_test = test_data[1:,0].astype(int)
    
    
    Y_train_one = np.zeros((Y_train.shape[0], 10))
    Y_train_one[np.arange(Y_train.shape[0]),Y_train] = 1
    
    Y_test_one = np.zeros((Y_test.shape[0], 10))
    Y_test_one[np.arange(Y_test.shape[0]),Y_test] = 1
    
    
    data = np.vstack((X_train, X_test))
    target = np.vstack((Y_train_one, Y_test_one))
    partials = np.zeros_like(target)
    
    with open("datasets/transition.pkl", "rb") as f:
        transition = pickle.load(f)
        
    for i in range(target.shape[0]):
        #Pdb().set_trace()
        print(i)
        correct = int(target[i].argmax())
        for j in range(10):
            prob = transition[correct][j]
            if(flip(prob)):
                partials[i][j] = 1
            
    with open("datasets/"+"MNIST.mat.pkl", "wb") as f:
        pickle.dump(data, f)
        pickle.dump(partials, f)
        pickle.dump(target, f)
        
        
def remakeTransition(filename):
    with open("datasets/"+filename+".mat.pkl", "rb") as f:
        data = pickle.load(f)
        partials = pickle.load(f)
        print(partials.sum(axis=1).mean())
        target = pickle.load(f)
    
    partials = np.zeros_like(target)
    labels = partials.shape[1]
    #print(labels)
    #print(partials.shape)
    with open("datasets/"+"transition2_"+filename+".pkl", "rb") as f:
        transition = pickle.load(f)
        
    for i in range(target.shape[0]):
        #print(i)
        correct = int(target[i].argmax())
        for j in range(labels):
            prob = transition[correct][j]
            #print(partials.shape)
            if(flip(prob)):
                partials[i,j] = 1
    print(partials.sum(axis=1).mean())
    with open("datasets/"+filename+"_transition2.mat.pkl", "wb") as f:
        pickle.dump(data, f)
        pickle.dump(partials, f)
        pickle.dump(target, f)


#for i in ['lost','MSRCv2','BirdSong','Soccer Player']:
#    makeTransition(i)
#    remakeTransition(i)
#makeMNIST()
#makeTransition()      
#print(Y_train_one)


#def MNIST():
    

#def main():
#    MNIST()
    #for i in [4]:
    #    remakeCC("lost", i)
    #train_dataset, real_train_dataset, val_dataset, real_val_dataset, test_dataset, real_test_dataset, input_dim, output_dim = loadTrain("lost_4.mat", 0, 10)
    #print(val_dataset.labels.sum(axis=1))        
#if __name__ == "__main__":
#    main() 
    