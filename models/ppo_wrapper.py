import numpy as np 
from env.board import Board

from env.board import Board

class EnvironmentWrapper():

    def __init__(self):
        """
        Initializes the EnvironmentWrapper class.

        Attributes:
        - board (Board): a Board object representing the game board
        - observation_space_len (int): the length of the observation space
        - action_space_len (int): the length of the action space
        - prev_max_tile (int): the maximum tile value in the previous step
        - num_steps_max_tile_did_not_change (int): the number of steps the maximum tile value did not change
        """
        self.board = Board()
        self.observation_space_len = self.board.state.flatten().shape[0] * 18 
        self.action_space_len = 4
        self.prev_max_tile = 2
        self.num_steps_max_tile_did_not_change = 1

    def reset(self):
        """
        Resets the game board and environment variables.

        Returns:
        - state (numpy.ndarray): the flattened state of the game board
        """
        self.board = Board()
        self.board.init_board()
        self.prev_max_tile = 2
        self.num_steps_max_tile_did_not_change = 1
        state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
        state = np.eye(18)[state].flatten()
        return state
    
    def step(self, action, r1 = True):
        """
        Performs one step of the game.

        Args:
        - action (int): the action to be taken
        - r1 (bool): flag for whether to use reward function 1

        Returns:
        - next_state (numpy.ndarray): the flattened state of the game board after the step
        - reward (float): the reward obtained from the step
        - done (bool): whether the game has ended
        - None: empty info dictionary
        """
        current_score = self.board.score
        old_zeros = np.count_nonzero(self.board.state == 0)
        self.board.move(action)
        new_zeros = np.count_nonzero(self.board.state == 0)
        self.board.max_number = self.board.get_max()
        if self.board.max_number == self.prev_max_tile:
            self.num_steps_max_tile_did_not_change += 1 
        else:
            self.num_steps_max_tile_did_not_change = 1 

        def reward1():
            """
            Calculates reward function 1.

            Returns:
            - reward (float): the reward obtained from reward function 1
            """
            getHigherTiles = np.log2(self.board.max_number)
            moreQuickly = 1 / self.num_steps_max_tile_did_not_change
            withBigScoreGain = self.board.score - current_score
            return (getHigherTiles**(moreQuickly)) * withBigScoreGain

        def reward2():
            """
            Calculates reward function 2.

            Returns:
            - reward (float): the reward obtained from reward function 2
            """
            return self.board.score - current_score

        def reward3():
            """
            Calculates reward function 3.

            Returns:
            - reward (int): the reward obtained from reward function 3
            """
            return np.count_nonzero(self.board.state == 0)

        def reward_final():
            """
            Calculates the final reward as a combination of reward functions 2 and 3.

            Returns:
            - reward (float): the final reward obtained from combining reward functions 2 and 3
            """
            return reward3() * reward2()

        # Get the reward
        if r1:
            reward = reward1()
        else:
            reward = reward_final()

        # Get the next state
        next_state = np.log2(self.board.state, out=np.zeros_like(self.board.state), where=(self.board.state != 0)).reshape(-1).astype(int)
        next_state = np.eye(18)[next_state].flatten()

        # Get the done flag
        done = self.board.terminal

        # Return the next state, reward, and done flag
        return next_state, reward, done, None


