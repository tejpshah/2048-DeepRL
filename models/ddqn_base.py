import copy
import os
import random
import shutil
from collections import deque, namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {DEVICE} device')

class replay_buffer():
  def __init__(self, capacity, longterm = 0.1):
    """
    Parameters
    ----------
    capacity : int
      Capacity of the replay buffer
    longterm : float
      Percentage of the replay buffer that is saved to longterm memory
    """
    self.capacity = capacity
    self.replay_buffer = deque(maxlen=capacity)

    self.longterm = longterm
    self.longterm_capacity = int(capacity * longterm)
    self.longterm_buffer = deque(maxlen=self.longterm_capacity)
    
    self.count = 0

  def push(self, *args):
    #Save early transitions to longterm buffer
    if self.count < self.longterm_capacity * self.longterm:
      self.longterm_buffer.append(TRANSITION(*args))
    self.count += 1
    if self.count > int(self.capacity / 2):
      self.count = 0
    

    #save transition to replay buffer
    self.replay_buffer.append(TRANSITION(*args))

  def sample(self, batch_size):
    """
    Samples a random size N minibatch from replay memory
    """
    if len(self.replay_buffer) == self.capacity:
      long_mem_size = int(batch_size / 10)
      batch_size -= long_mem_size
      return random.sample(self.replay_buffer, batch_size) + random.sample(self.replay_buffer, long_mem_size)
    
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
    self.Q_net = self.build_Q_net(n_observations, n_actions, arch=arch, drop=drop, batch_norm=batch_norm)

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

class EnvironmentWrapper():

    def __init__(self):

        # Initialize the board
        self.board = Board()
        self.observation_space_len = self.board.state.flatten().shape[0]
        self.action_space_len = 4
        self.steps_new_max = 0

    def reset(self):
        # Reset the board
        self.board.init_board()

        # Set the terminal flag to False
        self.board.terminal = False

        # Reset score back to 0
        self.board.score = 0

        # Reset steps_new_max
        self.steps_new_max = 0

        # Return the state and flatten it
        # so that it can be passed to the 
        # #network as a vector
        state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
        state = np.eye(18)[state].flatten()
        return state
    
    def step(self, action):

        # See current score 
        current_score = self.board.score

        # See current max
        current_max = self.board.get_max()

        # See current state
        # current_state = self.board.state.flatten()
        # print(self.board.state)
        current_state = self.board.get_state().copy()
        # current_state =np.log2(current_state, out=np.zeros_like(current_state), where=(current_state != 0)).reshape(-1).astype(int)
        # current_state = (np.eye(18)[current_state]).flatten()

        # Move the board
        self.board.move(action)

        # Get the next state
        # next_state = self.board.state.flatten()
        next_state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
        next_state = (np.eye(18)[next_state]).flatten()

        # Get the reward
        reward = self.get_reward(current_score, self.board.score, current_state, self.board.state, current_max, self.board.get_max())
        # print(reward)
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
      empty_spots = np.count_nonzero(next_state == 0)
      reward = empty_spots
      return reward
    
if __name__ == "__main__":
  from env.board import Board
  is_ipython = 'inline' in matplotlib.get_backend()
  if is_ipython: from IPython import display
  plt.ion()
  
  SAVE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Checkpoints',
                            'Test4_5_active.pt')
  TRANSITION = namedtuple('Transition',
                           ('state', 'action', 'reward', 'next_state'))

  env = EnvironmentWrapper()

  # Hyperparameters
  CAPACITY = 50000
  BATCH_SIZE = 128
  CLIPPING = 1000
  EPS_START = 0.02
  EPS_END = 0.01
  EPS_DECAY = 10000
  GAMMA = 0.99
  TAU = 0.001
  LR = 1e-5

  # Get the number of actions and the number of observations
  n_actions = env.action_space_len
  state = env.reset()
  n_observations = len(state)

  def load_params(net):
    if (os.path.exists(SAVE_PATH) and os.path.getsize(SAVE_PATH) > 0 ):
      print('Previous weights found, loading weights...')
      net.load_state_dict(torch.load(SAVE_PATH))
       
      #if loading weights succesdful, make a backup
      shutil.copy(SAVE_PATH, SAVE_PATH[0:-3] + 'BackUp.pt')
    else:
      #if no weights are found, create a file to indicate that no weights are found
      print('No weights found')

  # Initialize the networks
  Q_online = DoubleDQN(n_observations=n_observations, n_actions=n_actions, arch=(1, 256)).to(DEVICE)
  load_params(Q_online)
  Q_target = copy.deepcopy(Q_online).to(DEVICE)


  optimizer = optim.AdamW(Q_online.parameters(), lr=LR, amsgrad=True) 
  replay = replay_buffer(CAPACITY)

  steps_done = 0

  prev_state = None
  no_change_count = 0
  def select_action(state):
    global steps_done
    steps_done += 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

    global prev_state
    global no_change_count
    if prev_state is not None and (prev_state == state).all():
      no_change_count += 1
    else:
      prev_state = state.clone().detach()
      no_change_count = 0

    if np.random.uniform() > eps_threshold and no_change_count < 5:
      with torch.no_grad():
        return Q_online(state).max(1)[1].view(1, 1)
    else:
      return torch.tensor([[np.random.choice(n_actions)]], device=DEVICE, dtype=torch.long)

  episode_scores = []

  def plot_scores(show_results=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float32)
    if show_results:
      plt.title('Results')
    else:
      plt.clf()
      plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy(), label='Scores')
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
      mean_scores = scores_t.unfold(0, 100, 1).mean(1).view(-1)
      mean_scores = torch.cat((torch.zeros(99), mean_scores))
      plt.plot(mean_scores.numpy(), label='score average')
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

    #get mask of non final states 
    non_done_masks = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
    
    #get non final next states
    non_done_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the action already taken
    state_action_values = Q_online(state_batch).gather(dim=1, index=action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
      # Compute V(s_{t+1}) for all next states.
      next_best_actions = Q_online(non_done_next_states).argmax(dim=1).unsqueeze(1)
      next_state_values[non_done_masks] = Q_target(non_done_next_states).gather(dim=1, index=next_best_actions).squeeze(1)

      expected_state_action_values = reward_batch + GAMMA * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(Q_online.parameters(), CLIPPING)
    optimizer.step()


  """Training loop"""
  num_episodes = 10000

  for i_episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    for t in count():
      action = select_action(state)

      next_state, reward, done = env.step(action.item())

      reward = torch.tensor([reward], device=DEVICE, dtype=torch.float32)
      if not done:
        next_state = torch.tensor(next_state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
      else:
        next_state = None

      replay.push(state, action, reward, next_state)

      state = next_state

      optimize_model()

      Q_target_state_dict = Q_target.state_dict()
      Q_online_state_dict = Q_online.state_dict()
      for key in Q_online_state_dict:
        Q_target_state_dict[key] = TAU * Q_online_state_dict[key] + Q_target_state_dict[key] * (1.0 - TAU)
        Q_target.load_state_dict(Q_target_state_dict)

      if done:
        episode_scores.append(env.get_score())
        if i_episode % 10 == 0:
          print(f'Episode {i_episode} finished after {t+1} steps with score {env.get_score()}')
          torch.save(Q_online.state_dict(), SAVE_PATH)
        plot_scores()
        break


  print('Complete')
  plot_scores(show_results=True)
  plt.ioff()
  plt.show()
else:
  pass


#Reward functions:
      # current_grid = [((np.argwhere(current_state == 2**i)))
      #                 for i in range(1, 17)]
      # area = []
      # for inds in current_grid:
      #   a = 0
      #   if len(inds) > 2:
      #     x, y = inds[:, 0], inds[:, 1]
      #     a = 1/2 * np.sum(x * np.roll(y, 1) - np.roll(x, 1) * y)
      #   area.append(a)
      # area = sum(area)

      # merged = np.count_nonzero(next_state == 0) + 1 - np.count_nonzero(current_state == 0)
      # empty_spots = np.count_nonzero(next_state == 0)
# if self.board.terminal:
      #   return -100
      # delta_score = next_score -  current_score
      # merged = np.count_nonzero(next_state == 0) + 1 - np.count_nonzero(current_state == 0)
      # empty_spots = np.count_nonzero(next_state == 0)
      # reward = delta_score + merged * 100 + empty_spots * 10
      # return reward

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