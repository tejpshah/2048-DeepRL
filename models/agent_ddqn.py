import numpy as np
import os
import shutil
import torch
from torch import nn
from models.agent_random import AgentRandom
from models.env.board import Board
import torch.optim as optim
import copy

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

class AgentDDQN():
    
    def __init__(self, epsilon=1.0, arch=2, drop=False, batch_norm=False, steps=3):
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
      
      #creates Q network and loads/backsup weights if they exist
      self.Q = nn.Sequential(nn.Flatten())
      self.Q.add_module("Linear", self.build_Q_net(arch=arch, drop=drop, batch_norm=batch_norm))
      self.save_path = os.path.dirname(__file__)
      if (os.path.exists(self.save_path + "/data/Checkpoints/agent_ddqn.pt") 
        and os.path.getsize(self.save_path + "/data/Checkpoints/agent_ddqn.pt") > 0 ):
        shutil.copy(self.save_path + "/data/Checkpoints/agent_ddqn.pt", self.save_path + "/data/Checkpoints/agent_ddqnBackUp.pt")
        self.Q.load_state_dict(torch.load(self.save_path + "/data/Checkpoints/agent_ddqn.pt"))
      else:
        with open(self.save_path + "/data/Checkpoints/agent_ddqn.pt", "w") as f:
          pass
      
      #creates Q_target network and loads same weights as Q
      self.Q_target = nn.Sequential(nn.Flatten())
      self.Q_target.add_module("Linear", self.build_Q_net(arch=arch, drop=drop, batch_norm=batch_norm))
      self.Q_target.load_state_dict(self.Q.state_dict())
      
      self.env = Env_Emulator()  
      
      self.optimizer = optim.SGD(self.Q.parameters(), lr=0.001, momentum=.9) 
      
      self.steps = steps
      self.step_count = 0

    def choose_action(self, board, state=None):
      """
      chooses an action at random with probability = epsilon
      chooses optimal action with probability = 1 - epsilon
      
      Parameters
      ----------
      state : numpy array
        The state of the board
      """
      #decay epsilon
      self.epsilon = self.epsilon * 0.995 if self.epsilon > 0.1 else self.epsilon

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

      #emulates the action chosen
      self.env.step(board, action)

      self.step_count += 1

      #Learn avery # steps
      if self.step_count % self.steps == 0:

        states, actions, rewards, next_states, dones = self.env.sample()

        Q_exp = self.Q(states).gather(1, actions.view(-1, 1))

        #get best action from Q for next states
        next_actions = self.Q(next_states).detach().argmax(dim=1)
        next_actions_t = torch.LongTensor(next_actions).reshape(-1, 1).to(DEVICE)

        #get Q_target values for next states
        Q_target = self.Q_target(next_states).gather(1, next_actions_t)
        Q_target = rewards + 0.9 * (Q_target * (1 - dones))

        self.optimizer.zero_grad()
        Loss = nn.MSELoss()
        loss = Loss(Q_exp, Q_target)
        loss.backward()
        self.optimizer.step()
      
        #updates Q_target weights every # steps and saves weights
        self.Q_target.load_state_dict(self.Q.state_dict())
        torch.save(self.Q_target.state_dict(), self.save_path + '/data/Checkpoints/agent_ddqn.pt')
        self.step_count = 0

      return action
    

    def build_Q_net(self, arch:int, drop=False, batch_norm=False):
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
      layers = []
      for _ in range(arch + 1):
        layers.append(nn.Linear(16, 16))

        if batch_norm:
          layers.append(nn.BatchNorm1d(16))
      
        layers.append(nn.ReLU())
        
        if drop:
          layers.append(nn.Dropout(.5))
      
      layers.append(nn.Linear(16, 4))
      return nn.Sequential(*layers)
    
class Env_Emulator():
  def __init__(self):
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
    #makes copy of board
    board = copy.deepcopy(board)

    #gets current info from board
    curr_score = board.score
    curr_max = board.get_max()
    curr_state = torch.tensor(board.get_state(), dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)

    #takes action
    board.move(action)

    #get new info from board
    done = board.terminal
    next_score = board.score
    next_max = board.get_max()
    next_state = torch.tensor(board.get_state(), dtype=torch.float32, requires_grad=True, device=DEVICE).reshape(1, 4, 4)

    reward = np.sqrt(2 * next_score - curr_score) * (next_max / 2048)

    self.replay_buffer.append((curr_state, action, reward, next_state, done))
    board = None

  def sample(self):
    """
    Samples a minibatch from replay memory
    """
    minibatch_size = torch.randint(1, len(self.replay_buffer), (1,)).item()
    minibatch_ind = np.random.choice(list(range(len(self.replay_buffer))), size=minibatch_size, replace=False)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for i in minibatch_ind:
      state, action, reward, next_state, done = self.replay_buffer[i]
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      next_states.append(next_state)
      dones.append(done)

    if 10000 < self.get_replay_buffer_length() and np.random.uniform() < 0.1:
      self.reset_buffer()

    return (
      torch.stack(states).reshape(len(states), 4, 4),
      torch.tensor(actions, dtype=torch.long, device=DEVICE),
      torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
      torch.stack(next_states).reshape(len(next_states), 4, 4),
      torch.tensor(dones, dtype=torch.int32, device=DEVICE).unsqueeze(1))
  
  
  def reset_buffer(self):
    self.replay_buffer = []

  def get_replay_buffer(self):
    return self.replay_buffer
  
  def get_replay_buffer_length(self):
    return len(self.replay_buffer)
