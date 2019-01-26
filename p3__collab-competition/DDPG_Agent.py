import torch
from torch import nn,optim
import torch.nn.functional as F
from collections import namedtuple,deque
import numpy as np
import random
import copy

from model import Actor,Critic

lr_act = 1e-4
lr_cri = 1e-3
Tau = 1e-3
Gamma = 0.99

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:
    def __init__(self,state_size,action_size,seed):
        
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        
        # Noise process
        self.noise = OrnsteinUhlenbeck(action_size,seed)
        
        # Actor
        self.actor_local = Actor(state_size,action_size,seed).to(device)
        self.actor_target = Actor(state_size,action_size,seed).to(device)
        self.act_opt = optim.Adam(self.actor_local.parameters(),lr=lr_act)
        
        #Critic 
        self.critic_local = Critic(state_size,action_size,seed).to(device)
        self.critic_target = Critic(state_size,action_size,seed).to(device)
        self.cri_opt = optim.Adam(self.critic_local.parameters(),lr=lr_cri)
        
        # Initialize target networks
        
        self.soft_update(self.actor_local,self.actor_target,1)
        self.soft_update(self.critic_local,self.critic_target,1)
    
    def act(self,state,noise=True):
        
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()
        
        if noise:
            action += self.noise.sample()
        
        return np.clip(action,-1,1)
    
    def reset(self):
        self.noise.reset()
    
    def learn(self,experiences,Gamma):
        
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states,actions,rewards,next_states,dones = experiences
        
        # Critic Loss
        
        next_actions = self.actor_target(next_states)
        
        Q_tar_vals = self.critic_target(next_states,next_actions)
       
        Q_tar = rewards + (Gamma * Q_tar_vals * (1-dones))
        
        Q_est = self.critic_local(states,actions)
        
        loss_cri = F.mse_loss(Q_est,Q_tar)
        
        self.cri_opt.zero_grad()
        loss_cri.backward()
        self.cri_opt.step()
        
        # Actor Loss
        
        pred_actions = self.actor_local(states)
        
        # Gradient ascent
        loss_act = -self.critic_local(states,pred_actions).mean()
        
        self.act_opt.zero_grad()
        loss_act.backward()
        self.act_opt.step()
        
        # update target networks
        
        self.soft_update(self.actor_local,self.actor_target,Tau)
        self.soft_update(self.critic_local,self.critic_target,Tau)
    
    def soft_update(self,local,target,Tau):
        
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for target_p,local_p in zip(target.parameters(),local.parameters()):
            target_p.data.copy_(Tau * local_p.data + (1-Tau) * target_p.data)

            
class REPLAYBUFFER:
    
    def __init__(self,buffer_size,batch_size,seed):
        
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience',field_names=['state','action','reward','next_state','done'])
    
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory,self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        return len(self.memory)
        

class OrnsteinUhlenbeck:
    
    def __init__(self,action_size,seed,sigma=0.2,theta=0.15,mu=0.):
        
        self.seed = random.seed(seed)
        self.sigma = sigma
        self.theta = theta
        self.action_size = action_size
        self.mu = mu
        
        self.state = np.ones(action_size) * self.mu
       
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_size) * self.mu
    
    def sample(self):
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state