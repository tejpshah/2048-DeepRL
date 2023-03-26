import numpy as np
import torch
from torch import nn
from agent_random import AgentRandom
from env.board import Board
import torch.optim as optim

class AgentDQN(nn.Module):
    
    def __init__(self, epsilon=0.75, arch=2, drop=False, batch_norm=False):
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
      arch : int
        integer specifying ammount of fully connected layers
      """
      super().__init__()
      self.epsilon = epsilon
      self.Q, self.Q_hat = self.Linear_blk(arch=arch, drop=drop, batch_norm=batch_norm)
      self.env = Env_Emulator()  
      self.optimizer = optim.SGD(self.Q.parameters(), lr=0.01) 

    def choose_action(self, state=None,):
      """
      chooses an action at random with probability = epsilon
      chooses optimal action with probability = 1 - epsilon
      
      Parameters
      ----------
      state : numpy array
        The state of the board
      """
      #chooses random action
      if (np.random.uniform() < self.epsilon):
        return super().choose_action(state)
      
      #converts state from numpy to torch tensor
      state = torch.tensor(state, dtype=torch.float32, requires_grad=True, device='cuda')

      #chooses the optimal action
      action = torch.argmax(self.Q(state))

      #emulates the action chosen
      self.env.step(state, action)

      if self.env.get_replay_buffer_length() % 3 == 0:
        replay_buffer = torch.Tensor(self.env.get_replay_buffer())
        minibatch = replay_buffer[torch.randint(0, replay_buffer.shape[0], 3)]
        y = torch.zeros(minibatch.shape)
        for i, s, a, r, s_prime, done in enumerate(minibatch):
          if not done:
            y[i] = r + 0.9 * torch.max(self.Q_hat(s_prime))[0]
          else:
            y[i] = r
        
      return action
    

    def Linear_blk(self, arch:int, drop=False, batch_norm=False):
      """
      Creates arch ammount of fully connected layers
      
      Parameters
      ----------
      arch : int
        Integer specifying ammount of fully connected layers
      drop : bool
        Default False, indicating if dropout is wanted
      batch_norm : bool
        Default False, indicating if batch normalization is wanted
      
      Returns
      -------
      tuples of nn.Sequentail() filled with linear network initalized with the
      same weights
      """
      layers1, layers2 = [nn.Flatten], [nn.Flatten]
      for _ in range(arch + 1):
        layers1.append(nn.Linear(16, 16))
        layers2.append(nn.Linear(16, 16))
        layers2[-1].weight = nn.Parameter(layers1[-1].weight.data.clone())

        if batch_norm:
          layers1.append(nn.BatchNorm1d(16))
          layers2.append(nn.BatchNorm1d(16))
        
        layers1.append(nn.ReLU())
        layers2.append(nn.ReLU())
        
        if drop:
          layers1.append(nn.Dropout(.5))
          layers2.append(nn.Dropout(.5))
      layers1.append(nn.Linear(16, 4))
      layers2.append(nn.Linear(16, 4))
      return (nn.Sequential(*layers1), nn.Sequential(*layers2))
    
class Env_Emulator():
  def __init__(self) -> None:
    self.replay_buffer = []

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
    #makes clone of board
    board = board.clone()

    #gets current info from board
    curr_score = board.score
    curr_max = board.get_max()
    curr_state = board.get_state()

    #takes action
    board.move(action)

    #get new info from board
    done = board.terminal
    next_score = board.score
    next_max = board.get_max()

    reward = np.log2((next_score - curr_score)**2 + (next_max - curr_max))

    self.replay_buffer.append((curr_state, action, reward, board.get_state(), done))

  def reset(self):
    self.replay_buffer = []

  def get_replay_buffer(self):
    return self.replay_buffer
  
  def get_replay_buffer_length(self):
    return len(self.replay_buffer)
  
    
    
    

        