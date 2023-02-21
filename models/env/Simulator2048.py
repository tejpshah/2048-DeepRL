import numpy as np 

class Game2048():

    def __init__(self):
        self.init_board()

    def init_board(self, rows=4, cols=4):
        '''initializes board of 0s with two random tiles'''
        self.state = np.zeros((rows,cols))
        self.init_tile()
        self.init_tile() 

    def init_tile(self, p=0.9):
        '''randomly spawns tile of 2 (p=.9) or 4 (p=.1) in free space'''
        new_spawn_spots = self.get_spawn_tile_locations()
        new_tile_idx = np.random.choice(new_spawn_spots.shape[0], size=1, replace=False)
        new_row, new_col = new_spawn_spots[new_tile_idx].squeeze()
        self.state[new_row, new_col] = 2 if np.random.random() < p else 4

    def get_spawn_tile_locations(self):
        '''returns all [i j] in the current state where a new tile can spawn.'''
        return np.argwhere(self.state == 0)

    def get_score(self):
        '''returns sum of log(tile values) in current game state'''
        return np.log2(self.state).sum() 

    def get_max(self):
        '''returns the max log(tile value) in current game state.'''
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

    def move_up(self):
        '''Shifts and merges all tiles in the up direction.'''

        # Loop over each column of the game board
        for col in range(self.state.shape[1]):
            col_arr = self.state[:, col]
            new_col_arr = np.zeros(col_arr.shape[0])
            new_col_arr_idx = 0

            # Loop over each element in the column
            for row in range(col_arr.shape[0]):

                if col_arr[row] != 0:

                    # If the current element is non-zero
                    if new_col_arr[new_col_arr_idx - 1] == col_arr[row] and new_col_arr_idx > 0:

                        # If it can be merged with the previous element, double the value of the previous element
                        new_col_arr[new_col_arr_idx - 1] *= 2

                    else:
                        # Otherwise, add the current element to the end of the new column array
                        new_col_arr[new_col_arr_idx] = col_arr[row]
                        new_col_arr_idx += 1

            # Update the column in the game board with the values in the new column array
            self.state[:, col] = new_col_arr

    def move_down(self):
        '''Shifts and merges all tiles in the down direction.'''
        self.state = np.rot90(self.state, 2)
        self.move_up()
        self.state = np.rot90(self.state, 2)

    def move_left(self):
        '''Shifts and merges all tiles in the left direction.'''
        # Rotate the game board 90 degrees counterclockwise and call move_up
        self.state = np.rot90(self.state, 3)
        self.move_up()
        self.state = np.rot90(self.state, 1)

    def move_right(self):
        '''Shifts and merges all tiles in the right direction.'''
        # Rotate the game board 90 degrees clockwise and call move_up
        self.state = np.rot90(self.state, 1)
        self.move_up()
        self.state = np.rot90(self.state, 3)

G1 = Game2048()
print(G1.state)
G1.move_down()
print(G1.state)


