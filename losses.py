import torch
from IPython.core.debugger import Pdb
import torch.nn.functional as F

epsilon = 1e-6

def log_sigmoid(x):
    return torch.clamp(x, max=0) - torch.log(torch.exp(-torch.abs(x)) + 1) + 0.5 * torch.clamp(x, min=0, max=0)

def iexplr_loss(output, target):
    prob = torch.softmax(output, dim=1).detach()
    log_prob = torch.log_softmax(output, dim=1)
    target_probs = (prob*target.float()).sum(dim=1)
    mask = ((target == 1) & (prob > epsilon))
    loss = (prob[mask]*log_prob[mask]/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
    return -loss

def old_iexplr_loss(output, target):
    output = torch.softmax(output, dim=1)
    prob = output.detach()
    target_probs = (prob*target.float()).sum(dim=1)
    mask = ((target == 1) & (prob > epsilon))
    loss = (prob[mask]*torch.log(output[mask])/ target_probs.unsqueeze(1).expand_as(mask)[mask]).sum() / mask.size(0)
    return -loss


def cc_loss(output, target):
    log_target_prob = F.log_softmax(output, dim = 1)
    log_max_prob,max_prob_index = log_target_prob.max(dim=1)
    
    exp_argument = log_target_prob - log_max_prob.unsqueeze(dim=1)
    summ = (target*torch.exp(exp_argument)).sum(dim=1)
    log_total_prob = log_max_prob + torch.log(summ + epsilon)
    loss_tensor = (-1.0*log_total_prob).mean(dim=-1)
    return loss_tensor


def old_cc_loss(output, target):
    output = F.softmax(output)
    batch_size = output.shape[0]
    loss = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), target.view(output.shape[0], output.shape[1], 1))
    loss = torch.log(loss+epsilon)
    loss = torch.sum(loss)
    loss = torch.div(loss, -batch_size)
    return loss

def naive_loss(output, target):
    log_prob = F.log_softmax(output,dim=-1) 
    total_log_prob = (target*log_prob).sum(dim=-1)
    avg_log_prob = total_log_prob/target.sum(dim=-1)
    loss =  -1*avg_log_prob.mean()
    return loss

def ce_loss(scores, target):
    #Pdb().set_trace()
    target_no = target.argmax(dim=1)
    return F.cross_entropy(scores, target_no)


def min_loss(output, target):
    #Pdb().set_trace()
    
    loss = torch.log_softmax(output, dim=-1)
    #loss = loss[target == 0] = -float('inf')
    mask = target == 0
    loss = loss.masked_fill(mask,-float('inf')).max(dim=1).values
    #loss[~mask] = -float('inf')
    #loss = torch.max(loss, dim = 1).values
    loss = -loss.mean()
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
    loss = (torch.softmax(q, dim=1)*mask*torch.log_softmax(p, dim=1)).sum(dim=1).mean()
    return -loss

def select_reward_function(p, q, mask):
    rew = (torch.softmax(q, dim=1)*torch.softmax(p, dim=1)*mask).sum(dim=1).mean()
    return -rew

def old_select_loss_function(p, q, mask):
    p = torch.softmax(p, dim=1)
    q = torch.softmax(q, dim=1)
    #loss = (q*torch.log(p+epsilon)+(1-q)*torch.log(1-p+epsilon)).sum(dim=1).mean()
    loss = (q*mask*torch.log(p+epsilon)).sum(dim=1).mean()
    return -loss

def old_select_reward_function(p, q, mask):
    p = torch.softmax(p, dim=1)
    q = torch.softmax(q, dim=1)
    rew = (q*p*mask).sum(dim=1).mean()
    return -rew

def sample_reward_function(p, q, a, mask):
    p_soft = torch.softmax(p, dim=1)
    #q_sig = torch.sigmoid(q)
    #p_logsoft = torch.log_softmax(p)
    #q_logsig = torch.log_sigmoid(q)
    
    rew = (a*p_soft*mask).sum(dim=1) + ((1-a)*(1-p_soft)*mask).sum(dim=1)
    lap = a*log_sigmoid(q) + (1-a)*mask*(log_sigmoid(-q))
    #lap = a*log_sigmoid(q) + (1-a)*mask*(-q + log_sigmoid(q))
    lap = lap.sum(dim=1)
    
    return -torch.mean(rew * lap)

def sample_loss_function(p, q, a, mask):
    
    #loss = torch.log((a*torch.softmax(p, dim=1)*mask).sum(dim=1) +epsilon)
    loss = cc_loss(p, a*mask)
    return -0.5*torch.mean(loss)

def old_sample_reward_function(p, q, a, mask):
    p = torch.softmax(p, dim=1)
    q = torch.sigmoid(q)
    rew = (a*p*mask).sum(dim=1) + ((1-a)*(1-p)*mask).sum(dim=1)
    lap = a*torch.log(q+ epsilon) + (1-a)*mask*torch.log(1-q+ epsilon)
    lap = lap.sum(dim=1)
    
    return -torch.mean(rew * lap)

def old_sample_loss_function(p, q, a, mask):
    p = torch.softmax(p, dim=1)
    q = torch.sigmoid(q)
    loss = torch.log((a*p*mask).sum(dim=1) +epsilon)
    return -0.5*torch.mean(loss)












