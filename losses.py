import torch
from IPython.core.debugger import Pdb
import torch.nn.functional as F

epsilon = 1e-6

def iexplr_loss(output, target):
    prob = output.detach()
    target_probs = (prob*target.float()).sum(dim=1)
    mask = ((target == 1) & (prob > epsilon))
    loss = (prob[mask]*torch.log(output[mask])/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
    return -loss

def cc_loss_stable(output, target):
    log_target_prob = F.log_softmax(output, dim = 1)
    log_max_prob,max_prob_index = log_target_prob.max(dim=1)
    log_total_prob = log_max_prob + torch.log((target*torch.exp(log_target_prob - log_max_prob.unsqueeze(dim=1))).sum(dim=1))
    loss_tensor = (-1.0*log_total_prob).mean(dim=-1)
    return loss_tensor

    


def cc_loss(output, target):
    output = F.softmax(output)
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.log(loss+epsilon)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def naive_loss(output, target):
    #Pdb().set_trace()
    log_prob = F.log_softmax(output,dim=-1) 
    total_log_prob = (target*log_prob).sum(dim=-1)
    avg_log_prob = total_log_prob/target.sum(dim=-1)
    loss =  -1*avg_log_prob.mean()
    return loss
    """
    batch_size = output.shape[0]
    #loss = torch.log(output + epsilon)
    normalize = torch.sum(target, dim = 1)
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1)).flatten()
    loss = loss/normalize
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss
    """

def ce_loss(scores, target):
    #Pdb().set_trace()
    target_no = target.argmax(dim=1)
    return F.cross_entropy(scores, target_no)


def min_loss(output, target):
    batch_size = output.shape[0]
    loss = output * target
    loss = torch.max(loss, dim = 1).values
    loss = torch.log(loss+epsilon)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def regularized_cc_loss(lambd, output, target):
    c = cc_loss(output, target)
    m = min_loss(output, target)
    return  c + lambd*m

def naive_reward(output, target):
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def select_loss_function(p, q, mask):
    #loss = (q*torch.log(p+epsilon)+(1-q)*torch.log(1-p+epsilon)).sum(dim=1).mean()
    loss = (q*mask*torch.log(p+epsilon)).sum(dim=1).mean()
    return -loss

def select_reward_function(p, q, mask):
    rew = (q*p*mask).sum(dim=1).mean()
    return -rew

def sample_reward_function(p, q, a, mask):
    rew = (a*p*mask).sum(dim=1) + ((1-a)*(1-p)*mask).sum(dim=1)
    lap = a*torch.log(q+ epsilon) + (1-a)*mask*torch.log(1-q+ epsilon)
    lap = lap.sum(dim=1)
    
    return -torch.mean(rew * lap)

def sample_loss_function(p, q, a, mask):
    loss = torch.log((a*p*mask).sum(dim=1) +epsilon)
    return -0.5*torch.mean(loss)
