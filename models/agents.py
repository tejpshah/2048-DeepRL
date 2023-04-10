import torch 
import numpy as np 
from train_ppo_base import ActorCritic

class AgentRandom():
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
    
    def choose_action(self, state=None):
        '''
        given an observation state, select an action.
        for the baseline random agent, no action is given.
        for all other agents, override this method to include observation.
        '''
        return np.random.choice(self.actions)

class AgentPPO():
    def __init__(self, obs_space_size=16, act_space_size=4, hidden_layer_size, num_shared_layers, activation_fnction, device, model_path = 'ppo_2048_model.th'):
        
        self.device = device
        model = ActorCritic(obs_space_size,act_space_size, hidden_layer_size, num_shared_layers, activation_function)
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))

    def choose_action(self, state):
        logits, _ = model(torch.tensor([state.flatten()], dtype=torch.float32, device=self.device))
        act = torch.argmax(logits, dim=1).item()
        return act 



