# Note: This file works
import numpy as np 
from env.board import Board

class EnvironmentWrapper():

    def __init__(self):

        # Initialize the board
        self.board = Board()
        self.observation_space_len = self.board.state.flatten().shape[0] * 18 
        self.action_space_len = 4

        # Allows for decay of rewards
        self.prev_max_tile = 2
        self.num_steps_max_tile_did_not_change = 1

    def reset(self):

        self.board = Board()

        # Reset the board
        self.board.init_board()
        
        # Reset environment variables 
        self.prev_max_tile = 2
        self.num_steps_max_tile_did_not_change = 1

        # Return the state and flatten it
        # so that it can be passed to the 
        # #network as a vector
        state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
        state = np.eye(18)[state].flatten()
        return state
    
    def step(self, action, r1 = True):

        # See current score 
        current_score = self.board.score

        # Move the board
        self.board.move(action)

        # Change the max tile
        self.board.max_number = self.board.get_max()

        # Update if max tile did not change
        if self.board.max_number == self.prev_max_tile:
            self.num_steps_max_tile_did_not_change += 1 
        else: self.num_steps_max_tile_did_not_change = 1 

        def reward1():
            getHigherTiles = np.log2(self.board.max_number)
            moreQuickly = 1 / self.num_steps_max_tile_did_not_change
            withBigScoreGain = self.board.score - current_score
            return ( getHigherTiles**(moreQuickly) ) * withBigScoreGain

        def reward2():
            return self.board.score - current_score

        # Get the next state
        next_state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
        next_state = np.eye(18)[next_state].flatten()

        def reward3():
            return np.count_nonzero(next_state == 0)

        # Get the reward
        reward = reward3()

        # Get the done flag
        done = self.board.terminal

        # Return the next state, reward, and done flag
        return next_state, reward, done, None


