import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
from itertools import count

import torch
from torch import nn 
from torch.nn import functional as F
import torch.optim as optim

from .env.board import Board

import os
import shutil
import copy

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = None

class replay_buffer():
  def __init__(self, capacity=5000):
    self.replay_buffer = deque(maxlen=capacity)

  def push(self, *args):
    self.replay_buffer.append(TRANSITION(*args))

    # """
    # Emulates taking a step in the game given an action and 
    # stores it in replay memory

    # Parameters
    # ----------

    # board : Board object
    #   A board object that contains all information of the game
    
    # action : int
    #   An integer representing the action taken
    # """
    # #makes copy of board
    # board_copy = copy.deepcopy(board)

    # #gets current info from board
    # prev_score = board_copy.score
    # prev_state = torch.tensor(board_copy.get_state(), dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)
    # prev_max = board_copy.get_max()

    # #takes action
    # board_copy.move(action)

    # #get new info from board
    # done = board_copy.terminal
    # next_score = board_copy.score
    # next_max = board_copy.get_max()
    # next_state = torch.tensor(board_copy.get_state(), dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)

    # reward = self.get_reward(prev_score, next_score, prev_state, next_state, prev_max, next_max, rf='2')

    # self.replay_buffer.append((prev_state, action, reward, next_state, done))

    # board_copy = None

  def sample(self, batch_size):
    """
    Samples a random size N minibatch from replay memory
    """
    return random.sample(self.replay_buffer, batch_size) 
    # curr_replay_len = len(self.replay_buffer)
    # minibatch_size = torch.randint(1, curr_replay_len, (1,)).item()
    # minibatch_ind = np.random.choice(list(range(curr_replay_len)), size=minibatch_size, replace=False)
    # states, actions, rewards, next_states, dones = [], [], [], [], []
    # for i in minibatch_ind:
    #   state, action, reward, next_state, done = self.replay_buffer[i]
    #   states.append(state)
    #   actions.append(action)
    #   rewards.append(reward)
    #   next_states.append(next_state)
    #   dones.append(done)

    # return (
    #   torch.stack(states).reshape(len(states), 4, 4),
    #   torch.tensor(actions, dtype=torch.long, device=DEVICE),
    #   torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
    #   torch.stack(next_states).reshape(len(next_states), 4, 4),
    #   torch.tensor(dones, dtype=torch.int32, device=DEVICE).unsqueeze(1))

  def get_reward_avg(self):
    return torch.mean(torch.stack([x.reward for x in self.replay_buffer]))

  def get_replay_buffer(self):
    return self.replay_buffer
  
  def get_replay_buffer_length(self):
    return len(self.replay_buffer)
  
class DoubleDQN(nn.Module):

  def __init__(self, arch=(2, 32), drop=False, batch_norm=False):
    super(DoubleDQN, self).__init__()
    self.arch = arch
    self.Q_net = nn.Sequential(nn.Flatten())
    self.Q_net.add_module("Linear", self.build_Q_net(arch=arch, drop=drop, batch_norm=batch_norm))
    self.load_params()

  def forward(self, x):
    return self.Q_net(x)

  def build_Q_net(self, arch , drop, batch_norm):
      """
      Creates arch ammount of fully connected layers
      
      Parameters
      ----------
      arch : tuple(int,int)
        Tuple tuple(int,int) specifying ammount of fully connected layers and neurons per layer
      drop : bool
        Default False, indicating if dropout is wanted
      batch_norm : bool
        Default False, indicating if batch normalization is wanted
      
      """
      layers = []
      cl, lw = arch
      layers.append(nn.Linear(16, lw))
      layers.append(nn.ReLU())
      for _ in range(cl):
        layers.append(nn.Linear(lw, lw))

        if batch_norm: layers.append(nn.BatchNorm1d(lw))
      
        layers.append(nn.ReLU())
        
        if drop: layers.append(nn.Dropout(.5))
      
      layers.append(nn.Linear(lw, 4))
      return nn.Sequential(*layers)
  
  def load_params(self):
    global SAVE_PATH
    SAVE_PATH = os.path.dirname(__file__) + f'/data/Checkpoints/DoubleDQN-FC{self.arch[0]}-N{self.arch[1]}'
    if (os.path.exists(SAVE_PATH + '.pt') and os.path.getsize(SAVE_PATH + '.pt') > 0 ):
       #load weights
      state_dict = torch.load(SAVE_PATH + '.pt')
      new_state_dict = OrderedDict()
      for key, value in state_dict.items():
        key = key[6:]
        new_state_dict[key] = value
      self.Q_net.load_state_dict(new_state_dict)
       
      #if loading weights succesdful, make a backup
      shutil.copy(SAVE_PATH + '.pt', SAVE_PATH + 'BackUp.pt')
    else:
      #if no weights are found, create a file to indicate that no weights are found
      with open(SAVE_PATH + '.pt', "w") as f:
        pass

class EnvironmentWrapper():

    def __init__(self):

        # Initialize the board
        self.board = Board()
        self.observation_space_len = self.board.state.flatten().shape[0]
        self.action_space_len = 4

    def reset(self):
        # Reset the board
        self.board.init_board()
        self.board.terminal = False

        # Return the state and flatten it
        # so that it can be passed to the 
        # #network as a vector
        return self.board.state.flatten()
    
    def step(self, action):

        # See current score 
        current_score = self.board.score

        # See current max
        current_max = self.board.get_max()

        # See current state
        current_state = self.board.state.flatten()

        # Move the board
        self.board.move(action)

        # Get the next state
        next_state = self.board.state.flatten()

        # Get the reward
        reward = self.get_reward(current_score, self.board.score, current_state, next_state, current_max, self.board.get_max(), rf='2')

        # Get the done flag
        done = self.board.terminal

        # Return the next state, reward, and done flag
        return next_state, reward, done

    def get_reward(self,  current_score, next_score,  current_state, next_state, current_max, next_max, rf):
      """calculates reward for taking a step in the game"""
      #if no change in state, reward is -1 
      if (current_state == next_state).all():
          return -1

      if rf == '0':
        reward = np.sqrt(2 * next_score -  current_score) * (next_max / 2048)
      elif rf == '1':
        reward = np.sqrt(2 * next_score -  current_score) * np.log2(next_max)
      elif rf == '2':
        delta_score = next_score -  current_score if next_score >  current_score else 1
        higher_max = next_max if next_max >  current_max else 1
        reward = np.log10(delta_score) + np.log2(higher_max)

      return reward

CAPACITY = 5000
TRANSITION = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
GAMMA = 0.99
TAU = 0.005
LR = 0.001
BATCH_SIZE = 100000

env = EnvironmentWrapper()
n_actions = 4
n_observation = 16

Q_online = DoubleDQN(arch=(2, 32), drop=False, batch_norm=False).to(DEVICE)
Q_target = copy.deepcopy(Q_online).to(DEVICE)


optimizer = optim.Adam(Q_online.parameters(), lr=LR) 
replay = replay_buffer(CAPACITY)

steps_done = 0

def select_action(state):
  global steps_done
  steps_done += 1
  eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
  if np.random.uniform() > eps_threshold:
    with torch.no_grad():
      return Q_online(state).max(1)[1].view(1, 1)
  else:
    return torch.tensor([[np.random.choice(n_actions)]], device=DEVICE, dtype=torch.long) 
  
def optimize_model():
  if replay.get_replay_buffer_length() < BATCH_SIZE:
    return
  transitions = replay.sample(BATCH_SIZE)
  batch = TRANSITION(*zip(*transitions))

  non_done_masks = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=DEVICE, dtype=torch.bool)
  non_done_next_states = torch.cat([s for s in batch.next_state if s is not None])
  
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)

  state_action_values = Q_online(state_batch).gather(1, action_batch)

  next_action_batch = Q_online(non_done_next_states).max(1)[1].unsqueeze(1)
  next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
  with torch.no_grad():
    next_state_values[non_done_masks] = Q_target(non_done_next_states).gather(1, next_action_batch).squeeze(1)

  expected_state_action_values = (next_state_values * GAMMA) + reward_batch

  criterion = nn.MSELoss()
  loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

  optimizer.zero_grad()
  loss.backward()

  torch.nn.utils.clip_grad_value_(Q_online.parameters(), 100)
  optimizer.step()


"""Training loop"""
num_episodes = 0

for i_episode in range(num_episodes):
  state = env.reset()
  state = torch.tensor(state, device=DEVICE, dtype=torch.float).unsqueeze(0)
  for t in count():
    action = select_action(state)
    next_state, reward, done = env.step(action.item())
    reward = torch.tensor([reward], device=DEVICE, dtype=torch.float)
    if not done:
      next_state = torch.tensor(next_state, device=DEVICE, dtype=torch.float).unsqueeze(0)
    else:
      next_state = None

    replay.push(state, action, reward, next_state)
    
    state = next_state

    optimize_model()

    if done:
      break

  if i_episode % 10 == 0:
    Q_target.load_state_dict(Q_online.state_dict())
    torch.save(Q_online.state_dict(), SAVE_PATH + '.pt')
    print(f'Episode {i_episode} finished after {t+1} steps')

if num_episodes > 0:
  print(f'reward_avg: {replay.get_reward_avg()}') 



# class AgentDDQN():
    
#     def __init__(self, arch=(2, 16), drop=False, batch_norm=False, train=False, steps=3):
#       """
#       initializes actions agents can take.
#       from Agent random
#       0 : up
#       1 : down
#       2 : left
#       3 : right
      
#       Parameters
#       ----------
#       epsilon : float
#         The probability that the agent chooses a random action
#       arch : tuple(int, int)
#         tuple of ints specifying (connected layers, neurons per layer)
#       drop : bool
#         Default False, indicating if dropout is wanted
#       batch_norm : bool
#         Default False, indicating if batch normalization is wanted
#       steps : int
#         Default 3, indicating how many steps to update the target network
#       """
#       super().__init__()
#       self.actions = np.array([0,1,2,3])
#       self.arch = arch
#       self.train = train
      
#       #creates Q network and loads/backsup weights if they exist

#       #if gpu is available
#       self.Q, self.Q_target = self.Q.to(DEVICE), self.Q_target.to(DEVICE)
      
#       self.steps = steps
#       self.step_count = 0
      
#       self.old_board = None

#     def choose_action(self, board, state=None):
#       """
#       chooses an action at random with probability = epsilon
#       chooses optimal action with probability = 1 - epsilon
      
#       Parameters
#       ----------
#       state : numpy array
#         The state of the board
#       """
#       if np.random.uniform() < self.epsilon:
#         #chooses random action
#         action = np.random.choice(self.actions)
#       else:
#         #converts state from numpy to torch tensor
#         state = torch.tensor(state, dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)

#         #else chooses the optimal action
#         self.Q.eval()
#         action = torch.argmax(self.Q(state)).item()
#         self.Q.train()

#       if self.train:
#         #decreases epsilon if its a new board
#         if self.old_board is not None and id(board) != id(self.old_board):
#           self.epsilon = max(self.epsilon * 0.995, .1)
#           self.old_board = board
#         elif self.old_board is None:
#           self.old_board = board

#         #emulates the action chosen
#         self.env.step(board, action)

#         self.step_count += 1

#         #Learn avery # steps
#         if self.env.get_replay_buffer_length() > 5:
#           states, actions, rewards, next_states, dones = self.env.sample()

#           Q_exp = self.Q(states).gather(1, actions.view(-1, 1))

#           #get best action from Q for next states
#           next_actions = self.Q(next_states).detach().argmax(dim=1)
#           next_actions_t = next_actions.reshape(-1, 1).type(torch.long).to(DEVICE)

#           #get Q_target values for next states)
#           Q_target = self.Q_target(next_states).gather(1, next_actions_t)

#           Q_target = rewards + 0.99 * Q_target * (1 - dones)

#           loss = nn.functional.mse_loss(Q_exp, Q_target)
          
#           self.optimizer.zero_grad()
#           loss.backward()
#           self.optimizer.step()

#           torch.save(self.Q.state_dict(), self.save_path + '.pt')

#           #updates Q_target weights every # steps and saves weights
#           if self.step_count % self.steps == 0:
#             self.Q_target.load_state_dict(self.Q.state_dict())

#           if self.step_count % 1000 == 0:
#             print(f'\nepsilon: {self.epsilon}')
#             print(f'loss: {loss}')
#             print(f'reward_avg: {self.env.get_reward_avg()}')

#       return action
    