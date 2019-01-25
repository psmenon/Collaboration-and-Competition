# Collaboration-and-Competition

## Introduction

In this project I trained two agents to control rackets to bounce a ball over a net.

![alt text](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Hence,the goal of each agent is to keep the ball in play for as long as possible.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic and is considered solved when the agents get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents): After each episode, we add up the rewards that each agent received to get a score for each agent. We then take the maximum of these 2 scores.

## Training in Linux

1. Download the environment:  

    - Linux: [click here]()
    
To train the agent on Amazon Web Services (AWS), and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the environment.

(1) Create and activate a Python 3.6 environment using Anaconda:
   
   	conda create --name name_of_environment python=3.6
	source activate name_of_environment

(2) Clone repository and install dependencies

```bash
git clone https://github.com/psmenon/Collaboration-and-Competition.git
cd python
pip install .
```

(3) Place the environment file Tennis_Linux.zip in the p3_collab-competition/ folder and unzip the file:

```bash
$ unzip Tennis_Linux.zip
```

(4)  Launch Jupyter notebook

```bash
$ jupyter notebook
```
## Files inside p3__collab-competition/

```bash
TD3_agent.py - contains the Agent .

model.py -  contains the Pytorch neural network (actor and critic).

Continuous_Control.ipynb -  contains the code.

checkpoint_Actor.pth -  contains the weights of the solved actor network

checkpoint_Critic1.pth - contains the weights of the solved critic1 network

checkpoint_Critic2.pth - contains the weights of the solved critic2 network

Report.pdf - the project report
```



