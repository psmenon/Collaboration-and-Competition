import torch
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import random

from DDPG_Agent import Agent,REPLAYBUFFER

Buffer_size = int(3e5)
Batch_size = 128
Update_Every = 2
Gamma = 0.99

class Maddpg:
    def __init__(self,state_size,action_size,num_agents,seed):
        
        self.seed = random.seed(seed)
        self.num_agents = num_agents
        
        self.memory = REPLAYBUFFER(Buffer_size,Batch_size,seed)
        
        self.agents = [Agent(state_size,action_size,seed) for _ in range(self.num_agents)]
        self.t_step = 0
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    def act(self,states):
        
        actions = [agent.act(state) for agent,state in zip(self.agents,states)]
        return actions
    
    def step(self,states,actions,rewards,next_states,dones):
        
        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            self.memory.add(state,action,reward,next_state,done)
        
        self.t_step = (self.t_step + 1) % Update_Every
        if self.t_step == 0:
            if len(self.memory) > Batch_size:
                for agent in self.agents:
                    experiences = self.memory.sample()
                    agent.learn(experiences,Gamma)