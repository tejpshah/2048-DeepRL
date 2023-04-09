import os
import numpy as np
import torch
from . import ddqn_base 

SAVE_PATH = os.path.dirname(os.path.abspath(__file__)) + '\data\Checkpoints\Test4_3_4000.pt'

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
        self.Q_net = ddqn_base.DoubleDQN(n_observations=288, n_actions=4, arch=(1,256))
    
    def choose_action(self, state=None):
        '''
        given an observation state, select an action.
        for the baseline random agent, no action is given.
        for all other agents, override this method to include observation.
        '''
        if np.random.uniform() < 0.05:
          return np.random.choice(self.actions) 
        else:
          state = np.log2(state, out=np.zeros_like(state), where=(state != 0)).reshape(-1).astype(int)
          state = np.eye(18)[state]
          state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
          return torch.argmax(self.Q_net(state)).item()
        
    def load_params(net):
      if (os.path.exists(SAVE_PATH) and os.path.getsize(SAVE_PATH) > 0 ):
        print('Weights found, loading weights...')
        net.load_state_dict(torch.load(SAVE_PATH))
      else:
        #if no weights are found, create a file to indicate that no weights are found
        raise Exception('No weights found')