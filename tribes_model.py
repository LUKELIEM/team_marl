import os
import random
import time
import platform
import torch
import gym
import numpy as np


class Policy(torch.nn.Module):
    """
    We implement a 3-layer convolutional network for a specific agent
    identified by agent_inx. We comment out the LSTM implementation for now!
    """

    def __init__(self, input_channels, num_actions, agent_idx=1):
        super(Policy, self).__init__()
        
        # Tribal parameters
        self.tribe = None
        self.color = None
        self.idx = agent_idx   # This allows multiple learning agents       
        
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0
        
        self.temperature = 1.0               # This is to adjust exploit/explore 
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        self.features = self._init_features()
        self.action_head = self._init_action_head()
        
        # Deactivate actor-critic (CNN-LSTM) for now
        # self.lstm = self._init_lstm()
        # self.action_head = self._init_action_head()
        # self.value_head = self._init_value_head()

        # episode history
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1
        self.tagged_hist = []       
        self.tag_hist = []
        self.US_hits = []
        self.THEM_hits = []
        

    def _init_features(self):
        
        layers = []
        
        # [1,4,10,20] input 3D array
        layers.append(torch.nn.Conv2d(self.input_channels,
                                      16, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,10,20] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,5,10] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=3, stride=1, padding=0))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,3,8] feature maps

        return torch.nn.Sequential(*layers)

    def _init_action_head(self):
        # input [1,384]
        return torch.nn.Linear(384, self.num_actions)   # output [1,8]
    
    """
    # Disable CNN-LSTM actor critic for now

    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)
    """

        
    # The weights should be allowed to be saved into and load from agent-indexed model files
    # e.g. agent-1-model.pkl, agent-2-model.pkl, etc.
    def save_weights(self):
        file_name = 'agent-'+str(self.idx)+'-model.pkl'
        torch.save(self.state_dict(), file_name)   

    def load_weights(self):
        file_name = 'agent-'+str(self.idx)+'-model.pkl'
        if not os.path.exists(file_name):
                raise ValueError('map not found: ' + file_name)
        self.load_state_dict(torch.load(file_name))
        
    
    def forward(self, inputs):
        x = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 384(16x3x8)

        """
        # Disable CNN-LSTM actor critic for now
        
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        

        value = self.value_head(x)
        return action, value, (hx, cx)       
        
        """
        probs = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        return probs

    # This method attach the agent to a tribe by tribal name and color
    def attach_to_tribe(self, tribe_name, tribe_color):
        self.tribe = tribe_name
        self.color = tribe_color

    # This method resets agent info 
    def reset_info(self):
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0
    
    # This method loadsagent info 
    def load_info(self, info):
        # laser parameters
        self.tagged, self.laser_fired, self.US_hit, self.THEM_hit = info
        
        # save in episode history (to be used in tribal reward calculation)
        self.tagged_hist.append(self.tagged)       
        self.tag_hist.append(self.laser_fired)
        self.US_hits.append(self.US_hit)
        self.THEM_hits.append(self.THEM_hit)

    # This method flush the agent's history at the end of a game episode    
    def clear_history(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.tagged_hist[:]   
        del self.tag_hist[:]
        del self.US_hits[:]
        del self.THEM_hits[:]

        

# Just a dumb random agent
class Rdn_Policy():
            
    def __init__(self):
        super(Rdn_Policy, self).__init__()
        
        # Tribal parameters
        self.tribe = None
        self.color = None
        
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0

    def select_action(self, state):
        return random.randrange(0, 8)
    
    # This method attach the agent to a tribe by tribal name and color
    def attach_to_tribe(self, tribe_name, tribe_color):
        self.tribe = tribe_name
        self.color = tribe_color
        
    # This method loads agent info 
    def load_info(self, info):
        # laser parameters
        self.tagged, self.laser_fired, self.US_hit, self.THEM_hit = info


import numpy as np


"""
In this file, we will implement the Tribe Class:
- Agents attached to the Tribe have a "Us" versus "Them" mentality.
- The Tribe awards agents with tribal reward based on Culture.
"""
class Tribe():
    
    color = None
    name = None
    culture = None
    members = []

    # This is run when the tribe is created
    def __init__(self, name='Vikings', color='red', culture={'name':'individualist'}, agents=[]):
        self.color = color
        self.name = name
        self.culture = culture
        self.members = agents
        
        for agent in self.members:
            agent.attach_to_tribe(name, color)
    
    def sum_rewards(self):
        # Add the rewards of the tribe members elementwise
        rewards = [0 for i in range(len(self.members[0].rewards))]
        for a in self.members:
            rewards = [rewards[j] + a.rewards[j] for j in range(len(rewards))]
        return rewards   # return the tribe's rewards 
    
    def sum_US_hits(self):
        # Add the num of friendly fires (fragging) by the tribe members elementwise
        US_hits = [0 for i in range(len(self.members[0].US_hits))]
        for a in self.members:
            US_hits = [US_hits[j] + a.US_hits[j] for j in range(len(US_hits))]
        return US_hits   # return the tribe's fragging incidents

    
    def tribal_awards(self, US_hits = None, THEM_hits = None, tag_hist=None):
        culture = self.culture['name']
        if culture is 'cooperative':
            coop_factor = self.culture['coop_factor']
            rewards_to_tribe = self.sum_rewards() 
            num_members = len(self.members)
            awards = [r/num_members*coop_factor for r in rewards_to_tribe]
        elif culture is 'individualist':
            rewards_to_tribe = self.sum_rewards() 
            awards = [0 for r in rewards_to_tribe]
        elif culture is 'no_fragging':
            penalty = self.culture['penalty']
            awards = [friendly_fire * penalty for friendly_fire in US_hits]
        elif culture is 'pacifist':
            penalty = self.culture['penalty']
            awards = [laser * penalty for laser in tag_hist]
        elif culture is 'warlike':
            penalty = self.culture['penalty']
            reward = self.culture['reward']
            awards = [friendly_fire * penalty + enemy_hit * reward \
                      for friendly_fire, enemy_hit in zip(US_hits,THEM_hits)]
        else:
            awards = None
        return awards