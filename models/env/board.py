import numpy as np 
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt

class Board():

    EMPTY_CELL_COLOR = '#9e948a'
    CELL_BACKGROUND_COLOR_DICT = {
        0.0: '#ffffff',
        2.0: '#eee4da',
        4.0: '#ede0c8',
        8.0: '#f2b179',
        16.0: '#f59563',
        32.0: '#f67c5f',
        64.0: '#f65e3b',
        128.0: '#edcf72',
        256.0: '#edcc61',
        512.0: '#edc850',
        1024.0: '#edc53f',
        2048.0: '#edc22e',
        'beyond': '#3c3a32'
    }

    MOVEMENT_DICT = {
        0: 'up',
        1: 'down',
        2: 'left',
        3: 'right',
        -1: 'no move'
    }
    
    VISUAL_X_COORD = 0

    def __init__(self):
        self.init_board()
        self.score = 0
        self.max_number = 0
        self.last_move = None
        self.terminal = False

    def init_board(self, rows=4, cols=4):
        '''initializes board of 0s with two random tiles'''
        self.state = np.zeros((rows,cols))
        self.init_tile()
        self.init_tile() 

    def init_tile(self, p=0.9):
        '''randomly spawns tile of 2 (p=.9) or 4 (p=.1) in free space, or returns if there is no free space'''
        new_spawn_spots = self.get_spawn_tile_locations()
        if len(new_spawn_spots) == 0:
            return
        new_tile_idx = np.random.choice(new_spawn_spots.shape[0], size=1, replace=False)
        new_row, new_col = new_spawn_spots[new_tile_idx].squeeze()
        self.state[new_row, new_col] = 2 if np.random.random() < p else 4

    def get_spawn_tile_locations(self):
        '''returns all [i j] in the current state where a new tile can spawn.'''
        return np.argwhere(self.state == 0)

    def get_tilesum(self):
        '''returns sum of tile values in current game state'''
        return self.state.sum() 

    def get_max(self):
        '''returns the max tile value in current game state.'''
        return np.max(self.state)
    
    def get_state(self):
        '''returns the current state of the game'''
        return self.state
    
    def get_score(self):
        '''returns the current score of the game'''
        return self.score

    def get_last_move(self):
        '''returns the last move made in the game'''
        return self.last_move

    def is_terminal_state(self):
        '''
        episode is over is board is full or 2048 is achieved. 
        returns 1 if terminal state, 0 otherwise. 
        '''
        if 2048 in self.state:
            return 1
        if len(self.get_spawn_tile_locations()) > 0:
            return 0
        else: return 1
        # Check if any combinations can be made if the board is full
        for i in range(3):
            for j in range(3):
                if (self.state[i][j] == self.state[i+1][j] or self.state[i][j] == self.state[i][j+1]):
                    return 0
        for i in range(3):
            if (self.state[i][3] == self.state[i+1][3]):
                return 0
        for j in range(3):
            if (self.state[3][j] == self.state[3][j+1]):
                return 0
        return 1

    def _move_up(self):
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

                        # Update the score after the merge
                        self.score += new_col_arr[new_col_arr_idx - 1]

                    else:

                        # Otherwise, add the current element to the end of the new column array
                        new_col_arr[new_col_arr_idx] = col_arr[row]
                        new_col_arr_idx += 1

            # Update the column in the game board with the values in the new column array
            self.state[:, col] = new_col_arr
        self.last_move = 0

    def _move_down(self):
        '''Shifts and merges all tiles in the down direction.'''
        self.state = np.rot90(self.state, 2)
        self._move_up()
        self.state = np.rot90(self.state, 2)
        self.last_move = 1

    def _move_left(self):
        '''Shifts and merges all tiles in the left direction.'''
        # Rotate the game board 90 degrees counterclockwise and call move_up
        self.state = np.rot90(self.state, 3)
        self._move_up()
        self.state = np.rot90(self.state, 1)
        self.last_move = 2

    def _move_right(self):
        '''Shifts and merges all tiles in the right direction.'''
        self.state = np.rot90(self.state, 1)
        self._move_up()
        self.state = np.rot90(self.state, 3)
        self.last_move = 3

    def move(self, action):
        '''move the game up/down/left/right'''
        # move the board up/down/left/right
        if action == 'W' or action == 0: self._move_up() 
        elif action =='S' or action == 1: self._move_down() 
        elif action == 'A' or action == 2: self._move_left()
        elif action == 'D' or action == 3: self._move_right()

        # spawn a new tile 
        self.init_tile() 

        # check to see if the game state is terminal
        if self.is_terminal_state():
            self.terminal = True 
            return self.score, self.get_max(), self.get_tilesum()

    def print_game(self):
        '''prints out the state of the game'''
        print("------------------------")
        print(f"Current Game Score: {self.score}")
        print("------------------------")
        print(self.state)

    def visualize_board(self):
        """
        Visualize the current state of the game board using a table with colored cells.

        The function generates a matplotlib figure containing a table with the current state of the game board,
        where each cell is colored according to the value of the corresponding number. The function also shows
        the current score, the maximum number reached so far, and the last move made by the player.
        """

        # Set the size and layout of the figure
        plt.rcParams['figure.figsize'] = [3.00, 3.00]
        plt.rcParams['figure.autolayout'] = True

        # Create the figure and the axis
        fig, ax = plt.subplots(facecolor='white')

        # Remove the axis ticks and labels
        ax.axis('off')

        # Create a Pandas DataFrame from the game state
        df = pd.DataFrame(self.state, columns=['0', '1', '2', '3'])

        # Create a table from the DataFrame and add it to the axis
        table = ax.table(cellText=df.values, loc='center', cellLoc='center')

        # Adjust the layout of the figure
        fig.tight_layout()

        # Set the color of each cell according to its value
        for i in range(4):
            for j in range(4):
                data = self.state[i][j]
                self.max_number = max(data, self.max_number)
                color = self.CELL_BACKGROUND_COLOR_DICT[data]
                table[(i, j)].set_facecolor(color)

        # Add text labels to the axis with additional information about the game state
        ax.text(self.VISUAL_X_COORD, .8, 'Current Score: ' + str(self.score), transform=ax.transAxes, color='black')
        ax.text(self.VISUAL_X_COORD, .7, 'Max Number: ' + str(self.max_number), transform=ax.transAxes, color='black')
        ax.text(self.VISUAL_X_COORD, .2, 'Last Move: ' + self.last_move, transform=ax.transAxes, color='black')

        # Show the figure
        plt.show()

    def visualize_board_save(self, num):
        """
        Visualize the current state of the game board using a table with colored cells and save the figure to a file.

        The function generates a matplotlib figure containing a table with the current state of the game board,
        where each cell is colored according to the value of the corresponding number. The function also shows
        the current score, the maximum number reached so far, and the last move made by the player. The figure is
        saved to a PNG file with the specified number appended to the filename.
        """
        
        # Set the size and layout of the figure
        plt.rcParams['figure.figsize'] = [3.00, 3.00]
        plt.rcParams['figure.autolayout'] = True

        # Create the figure and the axis
        fig, ax = plt.subplots(facecolor='white')

        # Remove the axis ticks and labels
        ax.axis('off')

        # Create a Pandas DataFrame from the game state
        df = pd.DataFrame(self.state, columns=['0', '1', '2', '3'])

        # Create a table from the DataFrame and add it to the axis
        table = ax.table(cellText=df.values, loc='center', cellLoc='center')

        # Adjust the layout of the figure
        fig.tight_layout()

        # Set the color of each cell according to its value
        for i in range(4):
            for j in range(4):
                data = self.state[i][j]
                self.max_number = max(data, self.max_number)
                color = self.CELL_BACKGROUND_COLOR_DICT[data]
                table[(i, j)].set_facecolor(color)

        # Add text labels to the axis with additional information about the game state
        ax.text(self.VISUAL_X_COORD, .8, 'Current Score: ' + str(self.score), transform=ax.transAxes, color='black')
        ax.text(self.VISUAL_X_COORD, .7, 'Max Number: ' + str(self.max_number), transform=ax.transAxes, color='black')
        ax.text(self.VISUAL_X_COORD, .2, 'Last Move: ' + self.MOVEMENT_DICT[self.last_move], transform=ax.transAxes, color='black')

        # Save the figure to a file with the specified number appended to the filename
        plt.rc('savefig', dpi=300)
        plt.savefig('figure' + str(num) + '.png')

        # Close the figure to free up memory
        plt.close()