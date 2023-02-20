import numpy as np 

class Game2048():

    '''INITIALIZATION FUNCTIONS'''

    def __init__(self):
        self.init_board()

    def init_board(self, rows=4, cols=4):
        '''initializes board of 0s with two random tiles'''
        self.state = np.zeros((rows,cols))
        self.init_tile()
        self.init_tile() 

    '''SPAWN TILES ON THE BOARD'''

    def get_spawn_tile_locations(self):
        '''returns all [i j] in the current state where a new tile can spawn.'''
        return np.argwhere(self.state == 0)

    def init_tile(self):
        '''randomly spawns tile of 2 (p=.9) or 4 (p=.1) in free space'''
        new_spawn_spots = self.get_spawn_tile_locations()
        new_tile_idx = np.random.choice(new_spawn_spots.shape[0], size=1, replace=False)
        new_row, new_col = new_spawn_spots[new_tile_idx].squeeze()
        self.state[new_row, new_col] = 2 if np.random.random() < 0.9 else 4

    '''STATE TRANSITION FUNCTIONS'''

    def shift_board_up(self):
        pass 

    def shift_board_down(self):
        pass 

    def shift_board_left(self):
        pass 

    def shift_board_right(self):
        pass 

    '''STATES & REWARDS FOR RL METHODS'''

    def get_reward(self):
        '''
        returns the reward of the current state of the game.
        the log_2(current max score) is the current reward. 
        '''
        return np.log2(np.max(self.state))
    
    def get_state(self):
        '''returns the current state of the game'''
        return self.state

    def is_terminal_state(self):
        '''
        episode is over is board is full or 2048 is achieved. 
        returns 1 if terminal state, 0 otherwise. 
        '''
        return 1 if (2048 in self.state or len(self.get_spawn_tile_locations()) == 0) else 0 





G1 = Game2048()
print(G1.state)


