import gym
import sys
import pickle
from collections import deque

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# Initialize agents parameters
#   9 agents - 8 learning agents, 0 trained agents, 1 random agents
num_learners = 8
num_trained = 0
num_rdn = 1
num_statics = num_trained + num_rdn
num_agents = num_learners + num_statics  

# Initialize environment
game = "Gather"
num_actions = 8                       # 8 actions in Gathering

#   Data structure for agents
agents = []
actions = []
log_probs = []
tags = []
rewards = []
optimizers = []

# Initialize training parameters
warm_start = False
num_frames = 4      # environ observation consists of a list of 4 stacked frames per agent
max_episodes = 5000
max_frames = 300
max_frames_ep = 0   # track highest number of frames an episode can last
# These trainer parameters works for Atari Breakout
gamma = 0.99  
lr = 1e-3
temp_start = 1.8  # Temperature for explore/exploit
temp_end = 1.0
log_interval = 50
save_interval = 500


def unpack_env_obs(env_obs):
    """
    Gathering is a partially-observable Markov Game. env_obs returned by GatheringEnv is a numpy 
    array of dimension (num_agent, 800), which represents the agents' observations of the game.

    The 800 elements (view_box) encodes 4 layers of 10x20 pixels frames in the format:
    (viewbox_width, viewbox_depth, 4).
    
    This code reshapes the above into stacked frames that can be accepted by the Policy class:
    (batch_idx, in_channel, width, height)
    
    """
    
    num_agents = len(env_obs)  # environ observations is a list of agents' observations
    
    obs = []
    for i in range(num_agents):
        x = env_obs[i]   # take the indexed agent's observation
        x = torch.Tensor(x)   # Convert to tensor
        
        # Policy is a 3-layer CNN
        x = x.view(1, 10, 20, -1)  # reshape into environment defined stacked frames
        x = x.permute(0, 3, 1, 2)  # permute to Policy accepted stacked frames
        obs.append(x)
        
    return obs  # return a list of Tensors


"""
For now, we do not implement LSTM            
# LSTM Change: Need to cycle hx and cx thru function
def select_action(model, state, lstm_hc, cuda):
    hx , cx = lstm_hc 
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    if cuda:
        state = state.cuda()

    probs, value, (hx, cx) = model((Variable(state), (hx, cx)))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    # LSTM Change: Need to cycle hx and cx thru function
    return action.data[0], log_prob, value, (hx, cx)
"""


def select_learner_action(model, obs, cuda):
    """
    This code expects obs to be an array of stacked frames of the following dim:
    (batch_idx, in_channel, width, height)
    
    This is inputted into model - the agent's Policy, which outputs a probability 
    distribution over available actions.
    
    Policy gradient is implemented using torch.distributions.Categorical. 
    """
    
    # Policy is a 3-layer CNN
    # _, num_frames, width, height = obs.shape
    # obs = torch.FloatTensor(obs.reshape(-1, num_frames, width, height))
    
    # Policy is a 2-layer NN for now
    # obs = obs.view(1, -1)
   
    if cuda:
        obs = obs.cuda()
      
    probs = model(obs)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)

    return action.item(), log_prob 


def load_info(agents, narrate=False):
    for i in range(num_agents):    
        agents[i].load_info(info[i])
        if narrate:
            if agents[i].tagged:
                print('frame {}, agent{} is tagged'.format(frame,i))
            if agents[i].laser_fired:
                print('frame {}, agent{} fires its laser'.format(frame,i))
                print('and hit {} US and {} THEM'.format(agents[i].US_hit, agents[i].THEM_hit))
    return


# The main code starts here!!!

# Cold start
if warm_start is False:
   
    # Initialize learner agents, then load static agents (trained followed by random)
    for i in range(num_learners):
        print("Learner agent {}".format(i))
        agents.append(Policy(num_frames, num_actions, i)) # No weights loaded for learning agent
        optimizers.append(optim.Adam(agents[i].parameters(), lr=lr))
        
        # set up optimizer - this works for Atari Breakout
        # optimizers.append(optim.RMSprop(agents[i].parameters(), lr=lr, weight_decay=0.1)) 
        
    for i in range(num_learners, num_learners+num_trained):
        print ("No trained agent exist yet!")
        raise
        """
        Disable for now! No trained model exist!!!
        agents.append(Policy(num_frames, num_actions, i))
        agents[i].load_weights()         # load weight for static agent        
        """
    for i in range(num_learners+num_trained, num_agents):
        print("Load random agent {}".format(i))
        agents.append(Rdn_Policy())

    
    # Initialize all agent data
    actions = [0 for i in range(num_agents)]
    log_probs = [0 for i in range(num_agents)]
    tags = [0 for i in range(num_agents)]
    rewards = [0 for i in range(num_agents)]

    # Keep track of rewards learned by learners
    episode_reward = [0 for i in range(num_learners)]   # reward for an episode
    running_reward = [None for i in range(num_learners)]   # running average
    running_rewards = [[] for i in range(num_learners)]   # history of running averages
    best_reward = [0 for i in range(num_learners)]    # best running average (for storing best_model)

    # This is to support warm start for training
    prior_eps = 0

# Warm start
if warm_start:
    print ("Cannot warm start")
    raise
    
    """
    # Disable for now!  Need to ensure model can support training on GPU and game playing
    # on both CPU and GPU.
    
    data_file = 'results/{}.p'.format(game)

    try:
        with open(data_file, 'rb') as f:
            running_rewards = pickle.load(f)
            running_reward = running_rewards[-1]

        prior_eps = len(running_rewards)

        model_file = 'saved_models/actor_critic_{}_ep_{}.p'.format(game, prior_eps)
        with open(model_file, 'rb') as f:
            # Model Save and Load Update: Include both model and optim parameters
            saved_model = pickle.load(f)
            model, optimizer = saved_model

    except OSError:
        print('Saved file not found. Creating new cold start model.')
        model = Policy(input_channels=num_frames, num_actions=num_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=lr,
                                      weight_decay=0.1)
        running_rewards = []
        prior_eps = 0
    """

# Establish tribal association
tribes = []
tribes.append(Tribe(name='Vikings',color='blue', agents=[agents[0], agents[1], agents[2]]))
tribes.append(Tribe(name='Saxons', color='red', agents=[agents[3], agents[4]]))
tribes.append(Tribe(name='Franks', color='purple', agents=[agents[5], agents[6], agents[7]]))
tribes.append(Tribe(name='Crazies', color='yellow', agents=[agents[8]]))   # random agents are crazy!!!

# 9 agents in 4 tribes, used map defined in default.txt
agent_colors = [agent.color for agent in agents]
agent_tribes = [agent.tribe for agent in agents]
env = GatheringEnv(n_agents=num_agents,agent_colors=agent_colors, agent_tribes=agent_tribes, map_name='default')    
   
    
cuda = torch.cuda.is_available()

if cuda:
    for i in range(num_learners):    # Learning agents need to utilize GPU
        agents[i].cuda()

        
for ep in range(max_episodes):
    
    # Anneal temperature from temp_start to temp_end
    for i in range(num_learners):    # For learning agents
        agents[i].temperature = max(temp_end, temp_start - (temp_start - temp_end) * (ep / max_episodes))

    env_obs = env.reset()  # Env return observations

    # For Debug only
    # print (len(env_obs))
    # print (env_obs[0].shape)
    
    # Unpack observations into data structure compatible with agent Policy
    agents_obs = unpack_env_obs(env_obs)

    for i in range(num_learners):    # Reset agent info - laser tag statistics
        agents[i].reset_info()   

    # For Debug only
    # print (len(agents_obs))
    # print (agents_obs[0].shape)
    
    """
    For now, we do not stack observations, and we do not implement LSTM
    
    state = np.stack([state]*num_frames)

    # LSTM change - reset LSTM hidden units when episode begins
    cx = Variable(torch.zeros(1, 256))
    hx = Variable(torch.zeros(1, 256))
    if cuda:
        cx = cx.cuda()
        hx = hx.cuda()
    """

    episode_reward = [0 for i in range(num_learners)]   # reward for an episode
    
    for frame in range(max_frames):

        """
        For now, we do not implement LSTM
        # Select action
        # LSTM Change: Need to cycle hx and cx thru select_action
        action, log_prob, value, (hx,cx)  = select_action(model, state, (hx,cx), cuda)        
        """

        for i in range(num_learners):    # For learning agents
            actions[i], log_probs[i] = select_learner_action(agents[i], agents_obs[i], cuda)
            if actions[i] is 6:
                tags[i] += 1   # record a tag for accessing aggressiveness
            agents[i].saved_actions.append((log_probs[i]))
            
            # Do not implement LSTM for now
            # actions[i].saved_actions.append((log_prob, value))
            
        for i in range(num_learners, num_learners+num_trained):
            print ("No trained agent exist yet!")
            raise
        for i in range(num_learners+num_trained, num_agents):   # For random agents
            actions[i] = agents[i].select_action(agents_obs[i])
            if actions[i] is 6:
                tags[i] += 1   # record a tag for accessing aggressiveness

        # For Debug only
        # if frame % 20 == 0:
        #    print (actions) 
        #    print (log_probs)
            
        # Perform step        
        env_obs, reward, done, info = env.step(actions)
        
        """
        For Debug only
        print (env_obs)
        print (reward)
        print (done) 
        """
       
        # Unpack observations into data structure compatible with agent Policy
        agents_obs = unpack_env_obs(env_obs)
        load_info(agents, narrate=False)   # Load agent info for AI agents

        # For learner agents only, generate reward statistics and reward stack for policy gradient
        for i in range(num_learners):
            agents[i].rewards.append(reward[i])  # Stack rewards (for policy gradient)
            episode_reward[i] += reward[i]   # accumulate episode reward 
            
        """
        For now, we do not stack observation, may come in handy later on
        
        # Evict oldest diff add new diff to state
        next_state = np.stack([next_state]*num_frames)
        next_state[1:, :, :] = state[:-1, :, :]
        state = next_state
        """

        if any(done):
            print("Done after {} frames".format(frame))
            break
            
    if frame > max_frames_ep:
        max_frames_ep = frame    # Keep track of highest frames/episode

    # Update reward statistics for learners
    for i in range(num_learners):
        if running_reward[i] is None:
            running_reward[i] = episode_reward[i]
        running_reward[i] = running_reward[i] * 0.99 + episode_reward[i] * 0.01
        running_rewards[i].append(running_reward[i])

    # Track Episode #, temp and highest frames/episode
    if (ep+prior_eps+1) % log_interval == 0: 
        verbose_str = 'Episode {} complete'.format(ep+prior_eps+1)
        # verbose_str += '\tTemp = {:.4}'.format(model.temperature)
        # verbose_str += '\tMax frames = {}'.format(max_frames_ep+1)
        print(verbose_str)
    
        # Display rewards and running rewards for learning agents
        for i in range(num_learners):
            verbose_str = 'Learner:{}'.format(i)
            verbose_str += '\tReward total:{}'.format(episode_reward[i])
            verbose_str += '\tRunning mean: {:.4}'.format(running_reward[i])
            print(verbose_str)
    
    # Update model
    total_norm = finish_episode(agents[0:num_learners], optimizers[0:num_learners], gamma, cuda)

    if (ep+prior_eps+1) % log_interval == 0: 
        verbose_str = 'Max Norm = {}'.format(total_norm)    # Keep track of highest frames/episode
        print(verbose_str)
        
    if (ep+prior_eps+1) % save_interval == 0: 
        for i in range(num_learners):

            model_file = 'MA_models/MA{}_{}_ep_{}.p'.format(i, game, ep+prior_eps+1)
            data_file = 'results/MA{}_{}.p'.format(i, game)
            with open(model_file, 'wb') as f:
                # Model Save and Load Update: Include both model and optim parameters 
                pickle.dump((agents[i].cpu(), optimizers[i]), f)

            if cuda:
                agents[i] = agents[i].cuda()

            with open(data_file, 'wb') as f:
                pickle.dump(running_rewards[i], f)    
        
    """
    Do not save model for now!!!
    if (ep+prior_eps+1) % 500 == 0: 
        model_file = 'saved_models/actor_critic_{}_ep_{}.p'.format(
                                                                game,
                                                                ep+prior_eps+1)
        data_file = 'results/{}.p'.format(game)
        with open(model_file, 'wb') as f:
            # Model Save and Load Update: Include both model and optim parameters 
            pickle.dump((model.cpu(), optimizer), f)

        if cuda:
            model = model.cuda()

        with open(data_file, 'wb') as f:
            pickle.dump(running_rewards, f)    
    """

            
env.close()  # Close the environment
