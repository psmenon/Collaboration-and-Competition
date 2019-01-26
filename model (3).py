import torch
from torch import nn,optim
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hidden_init(layer):
    
    f_in = layer.weight.data.size(0)
    lim = 1. / np.sqrt(f_in)
    return (-lim,lim)

class Actor(nn.Module):
    def __init__(self,state_size,action_size,seed):
        super(Actor,self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        hidden_1 = 256
        hidden_2 = 128
        
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size,hidden_1)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        self.fc3 = nn.Linear(hidden_2,action_size)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,states):
        
        x = self.bn1(states)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

    
class Critic(nn.Module):
    def __init__(self,state_size,action_size,seed,dropout=0.2):
        super(Critic,self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        hidden_1 = 256
        hidden_2 = 128
        
        self.bn1 = nn.BatchNorm1d(state_size)
        
        self.fc1 = nn.Linear(state_size,hidden_1)
        self.fc2 = nn.Linear(hidden_1 + action_size,hidden_2)
        self.dropout = nn.Dropout(p = dropout)
        self.fc3 = nn.Linear(hidden_2,1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,states,actions):
        
        x = self.bn1(states)
        x = F.relu(self.fc1(x))
        
        xs = torch.cat((x,actions),dim=1)
        xs = F.relu(self.fc2(xs))
        
        return self.fc3(xs)
        