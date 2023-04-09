from env.board import Board

class EnvironmentWrapper():

    def __init__(self):

        # Initialize the board
        self.board = Board()
        self.observation_space_len = self.board.state.flatten().shape[0]
        self.action_space_len = 4

    def reset(self):

        self.board = Board()

        # Reset the board
        self.board.init_board()

        # Return the state and flatten it
        # so that it can be passed to the 
        # #network as a vector
        return self.board.state.flatten()
    
    def step(self, action):

        # See current score 
        current_score = self.board.score

        # Move the board
        self.board.move(action)

        # Get the reward
        scoreGain = self.board.score - current_score

        reward = scoreGain

        # Get the next state
        next_state = self.board.state.flatten()

        # Get the done flag
        done = self.board.terminal

        # Return the next state, reward, and done flag
        return next_state, reward, done, None