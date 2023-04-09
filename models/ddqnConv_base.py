import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque, OrderedDict
from itertools import count

import torch
from torch import nn 
import torch.optim as optim
from torch.nn import functional as F

import os
import shutil
import copy

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {DEVICE} device')
SAVE_PATH = None

class replay_buffer():
  def __init__(self, capacity):
    self.replay_buffer = deque(maxlen=capacity)

  def push(self, *args):
    self.replay_buffer.append(TRANSITION(*args))

  def sample(self, batch_size):
    """
    Samples a random size N minibatch from replay memory
    """
    return random.sample(self.replay_buffer, batch_size)

  def get_reward_avg(self):
    return torch.mean(torch.stack([x.reward for x in self.replay_buffer]))

  def get_replay_buffer(self):
    return self.replay_buffer
  
  def get_replay_buffer_length(self):
    return len(self.replay_buffer)
  
class DoubleDQN(nn.Module):

  def __init__(self, n_observations, n_actions, arch=(2, 32), drop=False, batch_norm=False):
    super(DoubleDQN, self).__init__()
    self.arch = arch
    self.Q_net = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                               nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                               nn.Flatten())
    self.Q_net.add_module('fc1', self.build_Q_net(32 * 4 * 4, n_actions, arch , drop, batch_norm))
    self.load_params()

  def forward(self, x):
    return self.Q_net(x)

  def build_Q_net(self, n_observations, n_actions, arch , drop, batch_norm):
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
      layers.append(nn.Linear(n_observations, lw))
      layers.append(nn.ReLU())
      for _ in range(cl):
        layers.append(nn.Linear(lw, lw))

        if batch_norm: layers.append(nn.BatchNorm1d(lw))
      
        layers.append(nn.ReLU())
        
        if drop: layers.append(nn.Dropout(.5))
      
      layers.append(nn.Linear(lw, n_actions))
      return nn.Sequential(*layers)
  
  def load_params(self):
    global SAVE_PATH
    SAVE_PATH = os.path.dirname(__file__) + f'/data/Checkpoints/DoubleDQN-FC{self.arch[0]}-N{self.arch[1]}'
    if (os.path.exists(SAVE_PATH + '.pt') and os.path.getsize(SAVE_PATH + '.pt') > 0 ):
       #load weights
      print('Previous weights found, loading weights...')
      state_dict = torch.load(SAVE_PATH + '.pt', map_location=DEVICE)
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

        # Set the terminal flag to False
        self.board.terminal = False

        # Reset score back to 0
        self.board.score = 0

        # Return the state and flatten it
        # so that it can be passed to the 
        # #network as a vector
        return self.board.state.reshape(1, 4, 4)
    
    def step(self, action):

        # See current score 
        current_score = self.board.score

        # See current max
        current_max = self.board.get_max()

        # See current state
        current_state = self.board.state.reshape(1, 4, 4)
        # Move the board
        self.board.move(action)

        # Get the next state
        next_state = self.board.state.reshape(1, 4, 4)

        # Get the reward
        reward = self.get_reward(current_score, self.board.score, current_state, next_state, current_max, self.board.get_max())

        # Get the done flag
        done = self.board.terminal

        # Return the next state, reward, and done flag
        return next_state, reward, done
    
    def get_score(self):
      return self.board.score
    
    def get_max(self):
      return self.board.get_max()

    def get_reward(self,  current_score, next_score,  current_state, next_state, current_max, next_max):
      """calculates reward for taking a step in the game"""
      # if self.board.terminal:
      #   return -100
      delta_score = next_score -  current_score
      merged = np.count_nonzero(next_state == 0) + 1 - np.count_nonzero(current_state == 0)
      empty_spots = np.count_nonzero(next_state == 0)
      reward = delta_score + merged * 100 + empty_spots * 10
      return reward

      # if self.board.terminal:
      #   return 100000 if next_max >= 2048 else -100
      # if (current_state == next_state).all():
      #   return -10
      
      # log_max = np.log2(next_max) * 10 if next_max > current_max else 0
      # delta_score = next_score -  current_score if next_score >  current_score else -.5
      # return delta_score + log_max
      
      #if no change in state, reward is -1 
      # if (current_state == next_state).all():
      #     return -1
      # if rf == '0':
      #   reward = np.sqrt(2 * next_score -  current_score) * (next_max / 2048)
      # elif rf == '1':
      #   reward = np.sqrt(2 * next_score -  current_score) * np.log2(next_max)
      # elif rf == '2':
      #   delta_score = next_score -  current_score if next_score >  current_score else 1
      #   higher_max = next_max if next_max >  current_max else 1
      #   reward = np.log10(delta_score) + np.log2(higher_max)
      # elif rf == "3":
      #   delta_score = next_score -  current_score if next_score >  current_score else 1
      #   reward = np.log10(delta_score)
      # elif rf == '4':
      #   reward = next_score - current_score
      # return reward

if __name__ == "__main__":
  from env.board import Board

  rng = np.random.default_rng()

  env = EnvironmentWrapper()

  is_ipython = 'inline' in matplotlib.get_backend()
  if is_ipython: from IPython import display
  plt.ion()
  
  # Hyperparameters
  CAPACITY = 100000
  BATCH_SIZE = 512
  UPDATE_EVERY = 3
  TRANSITION = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
  EPS_START = 0.90
  EPS_END = 0.05
  EPS_DECAY = 100000
  GAMMA = 0.99
  TAU = 0.005
  LR = 1e-4

  n_actions = env.action_space_len
  state = env.reset()
  n_observations = len(state)

  Q_online = DoubleDQN(n_observations=n_observations, n_actions=n_actions, arch=(2, 256)).to(DEVICE)
  Q_target = copy.deepcopy(Q_online).to(DEVICE)


  optimizer = optim.AdamW(Q_online.parameters(), lr=LR, amsgrad=True) 
  replay = replay_buffer(CAPACITY)

  steps_done = 0

  def select_action(state):
    global steps_done
    steps_done += 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    # if np.random.uniform() > eps_threshold:
    if rng.uniform() > eps_threshold:
      with torch.no_grad():
        return Q_online(state).max(1)[1].view(1, 1)
    else:
      # return torch.tensor([[np.random.choice(n_actions)]], device=DEVICE, dtype=torch.long)
      return torch.tensor([[rng.choice(n_actions)]], device=DEVICE, dtype=torch.long) 

  episode_scores = []
  episode_maxes = []

  def plot_scores(show_results=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    maxes_t = torch.tensor(episode_maxes, dtype=torch.float)
    if show_results:
      plt.title('Results')
    else:
      plt.clf()
      plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy(), label='Scores')
    plt.plot(maxes_t.numpy(), label='Maxes')
    # Take 100 episode averages and plot them too
    if len(maxes_t) >= 100:
      mean_scores = scores_t.unfold(0, 100, 1).mean(1).view(-1)
      mean_scores = torch.cat((torch.zeros(99), mean_scores))
      mean_maxes = maxes_t.unfold(0, 100, 1).mean(1).view(-1)
      mean_maxes = torch.cat((torch.zeros(99), mean_maxes))
      plt.plot(mean_scores.numpy(), label='score average')
      plt.plot(mean_maxes.numpy(), label='max average')
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
      if not show_results:
        display.display(plt.gcf())
        display.clear_output(wait=True)
      else:
        display.display(plt.gcf())

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

    next_best_actions = Q_online(non_done_next_states).max(1)[1].unsqueeze(1)
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
      next_state_values[non_done_masks] = Q_target(non_done_next_states).gather(1, next_best_actions).squeeze(1)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(Q_online.parameters(), 100)
    optimizer.step()


  """Training loop"""
  num_episodes = 5000

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

      if steps_done % UPDATE_EVERY == 0:
        Q_target_state_dict = Q_target.state_dict()
        Q_online_state_dict = Q_online.state_dict()
        for key in Q_online_state_dict:
          Q_target_state_dict[key] = TAU * Q_online_state_dict[key] + Q_target_state_dict[key] * (1.0 - TAU)
          Q_target.load_state_dict(Q_target_state_dict)
        
        #save model parametes
        torch.save(Q_online.state_dict(), SAVE_PATH + '.pt')

      if done:
        episode_scores.append(env.get_score())
        episode_maxes.append(env.get_max())
        if i_episode % 100 == 0:
          print(f'total steps: {steps_done}')
        plot_scores()
        break

  print('Complete')
  plot_scores(show_results=True)
  plt.ioff()
  plt.show()
else:
  pass