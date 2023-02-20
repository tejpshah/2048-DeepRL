import numpy as np 

class Game2048():
    def __init__(self):
        self.init_board()

    def init_board(self, rows=4, cols=4):
        '''initializes board of 0s with two random tiles'''
        self.state = np.zeros((rows,cols))
        self.init_tile()
        self.init_tile() 
        
    def get_spawn_tile_locations(self):
        '''returns all [i j] in the current state where a new tile can spawn.'''
        return np.argwhere(self.state == 0)

    def init_tile(self):
        '''randomly spawns tile of 2 (p=.9) or 4 (p=.1) in free space'''
        new_spawn_spots = self.get_spawn_tile_locations()
        new_tile_idx = np.random.choice(new_spawn_spots.shape[0], size=1, replace=False)
        new_row, new_col = new_spawn_spots[new_tile_idx].squeeze()
        self.state[new_row, new_col] = 2 if np.random.random() < 0.9 else 4

G1 = Game2048()
print(G1.state)


