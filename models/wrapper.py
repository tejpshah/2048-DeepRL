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

        old_zeros = np.count_nonzero(self.board.state == 0)

        # Move the board
        self.board.move(action)

        new_zeros = np.count_nonzero(self.board.state == 0)

        # Change the max tile
        self.board.max_number = self.board.get_max()

        # Update if max tile did not change
        if self.board.max_number == self.prev_max_tile:
            self.num_steps_max_tile_did_not_change += 1 
        else: self.num_steps_max_tile_did_not_change = 1 

        def reward1():
            getHigherTiles = np.log2(self.board.max_number)
            # getHigherTiles = self.board.max_number
            moreQuickly = 1 / self.num_steps_max_tile_did_not_change
            withBigScoreGain = self.board.score - current_score
            #withMoreTilesCombined = max(0,new_zeros - old_zeros)
            return ( getHigherTiles**(moreQuickly) ) * withBigScoreGain

        def reward2():
            return self.board.score - current_score

        def reward3():
            return np.count_nonzero(self.board.state == 0)

        def reward_final():
            return reward3() * reward2()

        def reward_function(game_state):
            # Constants to weight different aspects of the reward function
            lambda1 = 0.1
            lambda2 = 5
            lambda3 = 0.5

            # 1. Max tile value
            max_tile_value = np.log2(game_state.max())

            # 2. Number of empty cells
            num_empty_cells = sum([1 for row in game_state for cell in row if cell == 0])

            # 3. Smoothness of the board (sum of the absolute differences between neighboring cells)
            smoothness = 0
            for row in range(4):
                for col in range(4):
                    if row < 3:
                        smoothness -= abs(game_state[row][col] - game_state[row+1][col])
                    if col < 3:
                        smoothness -= abs(game_state[row][col] - game_state[row][col+1])

            # 4. Monotonicity (reward for having increasing or decreasing values along rows and columns)
            monotonicity = 0
            for row in range(4):
                row_diff = 0
                col_diff = 0
                for col in range(3):
                    if game_state[row][col] >= game_state[row][col+1]:
                        row_diff += 1
                    if game_state[col][row] >= game_state[col+1][row]:
                        col_diff += 1
                monotonicity += row_diff + col_diff

            # Combine the weighted components to form the final reward
            reward = (lambda1 * max_tile_value) + (lambda2 * num_empty_cells) + (lambda3 * smoothness) + (lambda3 * monotonicity)
            return reward

        # Get the reward
        reward = reward1()

        # Get the next state
        next_state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
        next_state = np.eye(18)[next_state].flatten()

        # Get the done flag
        done = self.board.terminal

        # Return the next state, reward, and done flag
        return next_state, reward, done, None


