import numpy as np 

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