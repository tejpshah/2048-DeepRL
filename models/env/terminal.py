import os
import numpy as np
from board import Board

class cli():
    def __init__(self):
        self.game = Board()

    def start(self):
        '''starts the game and allows user to plauy'''
        self.clear_screen()
        self.print_instructions()
        self.game.print_game()
        self.play_game()

    def clear_screen(self):
        '''clears the terminal screen'''
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_instructions(self):
        '''insturctions for the 2048 game'''
        print("Type in the terminal 'W'/'S'/'A'/'D' to move Up, Down, Left, or Right respectively. Press 'Q' to quit.")

    def play_game(self):
        '''allows user to lay games from terminal'''
        while not self.game.terminal:
            action = input()
            if action == 'Q': break 
            else: self.game.move(action)
            self.game.print_game()
            if self.game.terminal:
                print("Game over! Final score:", self.game.score)

if __name__ == "__main__":
    cli = cli()
    cli.start()
