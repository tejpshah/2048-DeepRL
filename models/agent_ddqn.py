import numpy as np
import os
import shutil
import torch
from torch import nn
from collections import deque
from models.agent_random import AgentRandom
from models.env.board import Board
import torch.optim as optim
import copy

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

class AgentDDQN():
    
    def __init__(self, epsilon=1.0, arch=(2, 16), drop=False, batch_norm=False, train=False, steps=3):
      """
      initializes actions agents can take.
      from Agent random
      0 : up
      1 : down
      2 : left
      3 : right
      
      Parameters
      ----------
      epsilon : float
        The probability that the agent chooses a random action
      arch : tuple(int, int)
        tuple of ints specifying (connected layers, neurons per layer)
      drop : bool
        Default False, indicating if dropout is wanted
      batch_norm : bool
        Default False, indicating if batch normalization is wanted
      steps : int
        Default 3, indicating how many steps to update the target network
      """
      super().__init__()
      self.actions = np.array([0,1,2,3])

      self.epsilon = epsilon
      self.arch = arch
      self.train = train
      
      #creates Q network and loads/backsup weights if they exist
      self.Q = nn.Sequential(nn.Flatten())
      self.Q.add_module("Linear", self.build_Q_net(arch=arch, drop=drop, batch_norm=batch_norm))
      
      self.save_path = os.path.dirname(__file__) + f'/data/Checkpoints/agent_ddqn{self.arch[0]}, {self.arch[1]}'
      if (os.path.exists(self.save_path + '.pt') and os.path.getsize(self.save_path + '.pt') > 0 ):
        self.Q.load_state_dict(torch.load(self.save_path + '.pt'))
        shutil.copy(self.save_path + '.pt', self.save_path + 'BackUp.pt')
        print('weights loaded')
      else:
        with open(self.save_path + '.pt', "w") as f:
          pass
      
      #creates Q_target network and loads same weights as Q
      self.Q_target = copy.deepcopy(self.Q)

      #if gpu is available
      self.Q, self.Q_target = self.Q.to(DEVICE), self.Q_target.to(DEVICE)
      
      self.env = Env_Emulator()  
      
      self.optimizer = optim.Adam(self.Q.parameters(), lr=0.001) 
      
      self.steps = steps
      self.step_count = 0
      
      self.old_board = None

    def choose_action(self, board, state=None):
      """
      chooses an action at random with probability = epsilon
      chooses optimal action with probability = 1 - epsilon
      
      Parameters
      ----------
      state : numpy array
        The state of the board
      """
      if np.random.uniform() < self.epsilon:
        #chooses random action
        action = np.random.choice(self.actions)
      else:
        #converts state from numpy to torch tensor
        state = torch.tensor(state, dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)

        #else chooses the optimal action
        self.Q.eval()
        action = torch.argmax(self.Q(state)).item()
        self.Q.train()

      if self.old_board is not None and id(board) != id(self.old_board):
        self.epsilon = max(self.epsilon * 0.999, .1)
        self.old_board = board
      elif self.old_board is None:
        self.old_board = board

      if self.train:
        #emulates the action chosen
        self.env.step(board, action)

        self.step_count += 1

        #Learn avery # steps
        if self.env.get_replay_buffer_length() > 10:
          states, actions, rewards, next_states, dones = self.env.sample()

          Q_exp = self.Q(states).gather(1, actions.view(-1, 1))

          #get best action from Q for next states
          next_actions = self.Q(next_states).detach().argmax(dim=1)
          next_actions_t = next_actions.reshape(-1, 1).type(torch.long).to(DEVICE)

          #get Q_target values for next states)
          Q_target = self.Q_target(next_states).gather(1, next_actions_t)

          Q_target = rewards + 0.99 * Q_target * (1 - dones)

          loss = nn.functional.mse_loss(Q_exp, Q_target)
          
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          torch.save(self.Q.state_dict(), self.save_path + '.pt')

          #updates Q_target weights every # steps and saves weights
          if self.step_count % self.steps == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

          if self.step_count % 2000 == 0:
            print(f'\nepsilon: {self.epsilon}')
            print(f'loss: {loss}')
            print(f'reward_avg: {rewards.mean()}')

      return action
    

    def build_Q_net(self, arch , drop=False, batch_norm=False):
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

        if batch_norm:
          layers.append(nn.BatchNorm1d(lw))
      
        layers.append(nn.ReLU())
        
        if drop:
          layers.append(nn.Dropout(.5))
      
      layers.append(nn.Linear(lw, 4))
      return nn.Sequential(*layers)
    
class Env_Emulator():
  def __init__(self, capacity=5000):
    self.replay_buffer = deque(maxlen=capacity)

  def step(self, board, action):
    """
    Emulates taking a step in the game given an action and 
    stores it in replay memory

    Parameters
    ----------

    board : Board object
      A board object that contains all information of the game
    
    action : int
      An integer representing the action taken
    """
    #makes copy of board
    board_copy = copy.deepcopy(board)

    #gets current info from board
    prev_score = board_copy.score
    prev_state = torch.tensor(board_copy.get_state(), dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)

    #takes action
    board_copy.move(action)

    #get new info from board
    done = board_copy.terminal
    next_score = board_copy.score
    next_max = board_copy.get_max()
    next_state = torch.tensor(board_copy.get_state(), dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)

    reward = self.get_reward(prev_score, next_score, prev_state, next_state, next_max, rf='2')

    self.replay_buffer.append((prev_state, action, reward, next_state, done))

    board_copy = None

  def sample(self):
    """
    Samples a random size N minibatch from replay memory
    """
    curr_replay_len = len(self.replay_buffer)
    minibatch_size = torch.randint(1, curr_replay_len, (1,)).item()
    minibatch_ind = np.random.choice(list(range(curr_replay_len)), size=minibatch_size, replace=False)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for i in minibatch_ind:
      state, action, reward, next_state, done = self.replay_buffer[i]
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      next_states.append(next_state)
      dones.append(done)

    return (
      torch.stack(states).reshape(len(states), 4, 4),
      torch.tensor(actions, dtype=torch.long, device=DEVICE),
      torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
      torch.stack(next_states).reshape(len(next_states), 4, 4),
      torch.tensor(dones, dtype=torch.int32, device=DEVICE).unsqueeze(1))
  
  def get_reward(self, prev_score, next_score, prev_state, next_state, next_max, rf):
    """calculates reward for taking a step in the game"""
    if rf == '0':
      if (prev_state == next_state).all():
        #if no change in state, reward is -1 
        reward = -1
      else:
        reward = np.sqrt(2 * next_score - prev_score) * (next_max / 2048)
    elif rf == '1':
      if (prev_state == next_state).all():
        #if no change in state, reward is -1 
        reward = -1
      else:
        reward = np.sqrt(2 * next_score - prev_score) * np.log2(next_max)
    elif rf == '2':
      if (prev_state == next_state).all():
        #if no change in state, reward is -1 
        reward = -1
      else:
        reward = np.log2(next_max)

    return reward

  
  def reset_buffer(self):
    self.replay_buffer = []

  def get_replay_buffer(self):
    return self.replay_buffer
  
  def get_replay_buffer_length(self):
    return len(self.replay_buffer)

