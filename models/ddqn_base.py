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

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {DEVICE} device')

class replay_buffer():
  def __init__(self, capacity, longterm = 0.1):
    """
    Initialize the replay buffer with given capacity and long-term memory percentage.
    
    Parameters
    ----------
    capacity : int
      Capacity of the replay buffer
    longterm : float
      Percentage of the replay buffer that is saved to longterm memory
    """
    self.capacity = capacity
    # Initialize the replay buffer with deque object, which is a double-ended queue that can add and remove elements from both ends
    self.replay_buffer = deque(maxlen=capacity)

    self.longterm = longterm
    # Calculate the maximum number of experiences that can be stored in long-term memory
    self.longterm_capacity = int(capacity * longterm)
    # Initialize the long-term memory with deque object
    self.longterm_buffer = deque(maxlen=self.longterm_capacity)

    # Initialize the counter for the number of experiences that have been added to the replay buffer to add to long-term memory periodically
    self.count = 0

  def push(self, *args):
    """
    Adds new experiences to the replay buffer and long-term memory
    """
    # Add experiences to the long-term buffer periodically
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
      # sameples batch_size experiences from replay buffer and long-term memory
      return random.sample(self.replay_buffer, batch_size) + random.sample(self.replay_buffer, long_mem_size)
    
    # sameples batch_size experiences from replay buffer
    return random.sample(self.replay_buffer, batch_size)


  def get_replay_buffer(self):
    """
    Returns the replay buffer.
    """
    return self.replay_buffer
  
  def get_replay_buffer_length(self):
    """
    Returns the length of the replay buffer.
    """
    return len(self.replay_buffer)
  
class DoubleDQN(nn.Module):

  def __init__(self, n_observations, n_actions, arch=(2, 32), drop=False, batch_norm=False):
    """
    Initializes the Double DQN model with the specified architecture.

    Parameters
    ----------
    n_observations : int
      Number of observations in the input state
    n_actions : int
      Number of possible actions in the output
    arch : tuple(int,int)
      Tuple of integers specifying the number of fully connected layers and the number of neurons per layer
    drop : bool
      Default False, indicating if dropout is wanted
    batch_norm : bool
      Default False, indicating if batch normalization is wanted
    """
    super(DoubleDQN, self).__init__()
    self.Q_net = self.build_Q_net(n_observations, n_actions, arch=arch, drop=drop, batch_norm=batch_norm)

  def forward(self, x):
    """
    Performs a forward pass through the Q_net.

    Parameters
    ----------
    x : tensor
      Input tensor of shape (batch_size, n_observations)

    Returns
    -------
    tensor
      Output tensor of shape (batch_size, n_actions)
    """
    return self.Q_net(x)

  def build_Q_net(self, n_observations, n_actions, arch , drop, batch_norm):
    """
    Builds the Q_net with the specified architecture.

    Parameters
    ----------
    n_observations : int
      Number of observations in the input state
    n_actions : int
      Number of possible actions in the output
    arch : tuple(int,int)
      Tuple of integers specifying the number of hidden layers and the number of neurons per hidden layer
    drop : bool
      Default False, indicating if dropout is wanted
    batch_norm : bool
      Default False, indicating if batch normalization is wanted

    Returns
    -------
    nn.Sequential
      Sequential model object containing the fully connected layers
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
  """
  A wrapper for the 2048 game board, to be used for reinforcement learning.
  """

  def __init__(self):

      # Initialize the board
      self.board = Board()

      # Set the size of the observation space
      state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
      state = np.eye(18)[state].flatten()
      self.observation_space_len = state.shape[0]

      # Set the size of the action space
      self.action_space_len = 4

  def reset(self):
      """
      Resets the game board and returns the initial state of the game.
      """
      # Reset the board
      self.board.init_board()

      # Set the terminal flag to False
      self.board.terminal = False

      # Reset score back to 0
      self.board.score = 0

      # Return the state and flatten it so that it can be passed to the network as a vector
      state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
      state = np.eye(18)[state].flatten()
      return state
  
  def step(self, action):
      """
      Takes a step in the game based on the chosen action and returns the new state,
      reward, and whether the game is over or not.
      """
      # Move the game board based on the chosen action
      self.board.move(action)

      # Get the next state of the game board
      next_state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
      next_state = (np.eye(18)[next_state]).flatten()

      # Get the reward for taking the step
      reward = self.get_reward(self.board.state)

      # Check if the game is over
      done = self.board.terminal

      # Return the next state, reward, and done flag
      return next_state, reward, done
  
  def get_score(self):
    """
    Returns the current score of the game.
    """
    return self.board.score

  def get_reward(self, next_state):
    """
    Calculates the reward for taking a step in the game based on the number of empty spots on the board.
    """
    empty_spots = np.count_nonzero(next_state == 0)
    reward = empty_spots
    return reward
    
if __name__ == "__main__":
  from env.board import Board
  is_ipython = 'inline' in matplotlib.get_backend()
  if is_ipython: from IPython import display
  plt.ion()
  
  # Define save path for model checkpoints
  SAVE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Checkpoints', 'Test4_5_active.pt')
  
  # Define named tuple for storing transition data
  TRANSITION = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

  # Initialize the environment
  env = EnvironmentWrapper()

  # Define hyperparameters
  CAPACITY = 50000
  BATCH_SIZE = 128
  CLIPPING = 1000
  EPS_START = 0.02
  EPS_END = 0.01
  EPS_DECAY = 10000
  GAMMA = 0.99
  TAU = 0.001
  LR = 1e-5

  # Get the number of actions and the number of observations from the environment
  n_actions = env.action_space_len
  state = env.reset()
  n_observations = len(state)

  # Define function for loading model weights
  def load_params(net):
    """
    Load the weights of the network from the checkpoint file if it exists and is not empty.
    Otherwise, create an empty file to indicate that no weights are found.
    """
    if (os.path.exists(SAVE_PATH) and os.path.getsize(SAVE_PATH) > 0 ):
      print('Previous weights found, loading weights...')
      net.load_state_dict(torch.load(SAVE_PATH))
       
      #if loading weights succesdful, make a backup
      shutil.copy(SAVE_PATH, SAVE_PATH[0:-3] + 'BackUp.pt')
    else:
      #if no weights are found, create a file to indicate that no weights are found
      print('No weights found')

  # Initialize the online and target Q-networks
  Q_online = DoubleDQN(n_observations=n_observations, n_actions=n_actions, arch=(1, 256)).to(DEVICE)
  load_params(Q_online)
  Q_target = copy.deepcopy(Q_online).to(DEVICE) 

  # Define optimizer and replay buffer
  optimizer = optim.AdamW(Q_online.parameters(), lr=LR, amsgrad=True) 
  replay = replay_buffer(CAPACITY)

  # Initialize step count
  steps_done = 0

  # Initialize variables for detecting when the agent is choosing an action with no change in state
  prev_state = None
  no_change_count = 0

  # Define function for selecting an action
  def select_action(state):
    global steps_done
    global prev_state
    global no_change_count
    
    # Increment step count
    steps_done += 1

    # Calculate exploration rate
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

    # Check if current state is the same as the previous state
    if prev_state is not None and (prev_state == state).all():
      no_change_count += 1
    else:
      prev_state = state.clone().detach()
      no_change_count = 0

    # Choose action based on epsilon-greedy policy
    if np.random.uniform() > eps_threshold and no_change_count < 5:
      with torch.no_grad():
        return Q_online(state).max(1)[1].view(1, 1)
    else:
      return torch.tensor([[np.random.choice(n_actions)]], device=DEVICE, dtype=torch.long)

  episode_scores = []

  # function to plot episode scores
  def plot_scores(show_results=False):
    plt.figure(1)

    # convert episode scores to a tensor
    scores_t = torch.tensor(episode_scores, dtype=torch.float32)

    # set title based on whether we are showing results or training
    if show_results:
      plt.title('Results')
    else:
      plt.clf()
      plt.title('Training...')

    # set x and y labels
    plt.xlabel('Episode')
    plt.ylabel('Score')

    # plot the episode scores
    plt.plot(scores_t.numpy(), label='Scores')

    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
      mean_scores = scores_t.unfold(0, 100, 1).mean(1).view(-1)
      mean_scores = torch.cat((torch.zeros(99), mean_scores))
      plt.plot(mean_scores.numpy(), label='score average')
    
    # pause a bit so that plots are updated
    plt.pause(0.001)

    # display the figure if using IPython
    if is_ipython:
      if not show_results:
        display.display(plt.gcf())
        display.clear_output(wait=True)
      else:
        display.display(plt.gcf())

  # function to optimize the model
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

    # compute loss using Smooth L1 loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # zero the gradients, backpropagate the loss, and clip the gradients
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(Q_online.parameters(), CLIPPING)
    optimizer.step()


  # set the number of episodes to train for
  num_episodes = 20000

  # run the training loop
  for i_episode in range(num_episodes):

    # reset environment and convert state to tensor
    state = env.reset()
    state = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0)

    #run episode
    for t in count():
      
      # select an action using the online Q network
      action = select_action(state)

      # take the action and observe the next state, reward, and done flag
      next_state, reward, done = env.step(action.item())

      # convert the reward to a tensor
      reward = torch.tensor([reward], device=DEVICE, dtype=torch.float32)

      # if the episode is not done, convert the next state to a tensor and add it to the replay buffer
      if not done:
        next_state = torch.tensor(next_state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
      else:
        next_state = None

      # add the current state, action, reward, and next state to the replay buffer
      replay.push(state, action, reward, next_state)

      # set the current state to the next state
      state = next_state

      # optimize the online Q network by sampling from the replay buffer
      optimize_model()

      # update the target Q network by copying the online Q network with a soft update
      Q_target_state_dict = Q_target.state_dict()
      Q_online_state_dict = Q_online.state_dict()
      for key in Q_online_state_dict:
        Q_target_state_dict[key] = TAU * Q_online_state_dict[key] + Q_target_state_dict[key] * (1.0 - TAU)
        Q_target.load_state_dict(Q_target_state_dict)

      # if the episode is done, record the score, save the online Q network, plot the scores, and break out of the loop
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