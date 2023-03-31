import numpy as np
import torch
from . import ddqn_base 

class AgentDoubleDQN():
    def __init__(self):
        '''
        initializes actions agents can take.
        include other standard components as we build base class. 
        all other agents should inherit from this class. 

        0 : up
        1 : down
        2 : left
        3 : right
        '''
        self.actions = np.array([0,1,2,3])
        self.Q_net = ddqn_base.DoubleDQN()
    
    def choose_action(self, state=None):
        '''
        given an observation state, select an action.
        for the baseline random agent, no action is given.
        for all other agents, override this method to include observation.
        '''
        if np.random.uniform() < 0.05:
          return np.random.choice(self.actions) 
        else:
          state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
          return torch.argmax(self.Q_net(state)).item()