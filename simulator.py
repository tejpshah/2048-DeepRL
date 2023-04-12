import time 
import multiprocessing as mp
from models.env.board import Board
from models.agent_ddqn import AgentDoubleDQN
from models.utils.plotting import Plotter
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch

class Simulator():

    VISUAL_X_COORD = 0

    def __init__(self, agent=None):

        # selects which agent to run simulations from
        self.agent =  AgentDoubleDQN()
        
        if torch.cuda.is_available(): mp.set_start_method('spawn')

        # stores simulation info for plotting
        self.game_scores = mp.Manager().dict()
        self.max_scores = mp.Manager().dict() 
        self.num_steps = mp.Manager().dict()

        # stores a tensor of the game states
        self.gameplay_tensor = None

        # a tensor with the score at the time and the last move
        # self.game_movements = torch.zeroes([1, 2])
    
    def run_episode(self):

        game = Board()

        n_steps = 0 

        current_game = torch.zeros(1, 4, 4)
        # current_game_stats = torch.zeros(1, 2)

        # runs an episode until termination 
        while not game.is_terminal_state():

            # gets state, and makes action based on state
            state = game.get_state() 

            #choose action
            action = self.agent.choose_action(state)

            # updates game board steps
            game.move(action)
            n_steps += 1

            # plot and save image of game at that time from board game function
            # game.visualize_board_save(n_steps)
            
        # updates simulation info dictionaries 
        self.num_steps[n_steps] = self.num_steps.get(n_steps, 0) + 1
        self.max_scores[game.get_max()] = self.max_scores.get(game.get_max(), 0) + 1
        self.game_scores[game.score] = self.game_scores.get(game.score, 0) + 1
    
    def run_episodes(self, num_episodes=100, num_procs=1):
        start = time.time()

        # divide episodes among processes
        episodes_per_proc = num_episodes // num_procs
        procs = []
        for _ in range(num_procs):
            proc = mp.Process(target=self.run_episodes_worker, args=(episodes_per_proc,))
            proc.start()
            procs.append(proc)

        # wait for processes to finish
        for proc in procs:
            proc.join()

        end = time.time()
        print(f"It took {end-start:.3f} seconds to run {num_episodes} simulations.")

    def run_episodes_worker(self, num_episodes):
        print(f"Starting {num_episodes} episodes...")
        for i in range(num_episodes):
            if i % 10 == 0:
                print(f"Episode: {i}")
            self.run_episode()

    def get_simulation_info(self):
        print('\nTHIS WAS THE SIMULATION INFO:')
        print(f"Game Scores Dictionary: {self.game_scores}")
        print(f"Max Scores Dictionary: {self.max_scores}")
        print(f"Episode Steps Dictionary: {self.num_steps}\n")
    
    def visualize_board_video(self, fn = 'video.mp4'):
        os.system('ffmpeg -r 3 -i figure%d.png -vcodec mpeg4 -y '+fn)
        print("A video showing the agent's traversal is ready to view. Opening...")
        os.system('open '+fn)
        # TODO remove the images after creating them
        # for i in range(self.max_num_steps):
            # os.remove('figure' + str(i+1) + '.png')
    
    # function for visualizing a single state and saving as a png
    def visualize_board_simulator_single(self, num, stateTensor):
        plt.rcParams['figure.figsize'] = [3.00, 3.00]
        plt.rcParams['figure.autolayout'] = True
        fig, ax = plt.subplots(facecolor ='white')
        ax.axis('off')
        df = pd.DataFrame(stateTensor.numpy(), columns = ['0', '1', '2', '3'])
        table = ax.table(cellText = df.values, loc = 'center', cellLoc='center')
        fig.tight_layout()
        max_number = 0
        for i in range(4):
            for j in range(4):
                data = stateTensor[i, j]
                max_number = max(data, max_number)
                color = Board.CELL_BACKGROUND_COLOR_DICT[data]
                table[(i, j)].set_facecolor(color)
        ax.text(self.VISUAL_X_COORD, .8, 'Current Score: ' + str(self.score), transform = ax.transAxes, color = 'black')
        ax.text(self.VISUAL_X_COORD, .7, 'Max Number: ' + str(self.max_number), transform = ax.transAxes, color = 'black')
        ax.text(self.VISUAL_X_COORD, .2, 'Last Move: ' + self.last_move, transform = ax.transAxes, color = 'black')
        plt.savefig(os.path.dirname(__file__) + 'data/Visual/' + 'figure' + str(num) + '.png')
        plt.close()

    # loop for visualizing all states of gameplay
    def visualize_gameplay(self, gameplayTensor):
        for i in range(gameplayTensor.size(dim=0)):
            self.visualize_board_simulator_single(i, gameplayTensor[i, :, :])
    
    #Plots and saves the Data
    def plt_sim(self, plt_type="histo"):
        p = Plotter(game_scores=self.game_scores, max_scores=self.max_scores,
                    num_steps=self.num_steps, agent=self.agent)
        p.save_json_info()


if __name__ == "__main__":
    S1 = Simulator()
    S1.run_episodes()
    # S1.visualize_board_video()
    # S1.get_simulation_info()
    S1.plt_sim()
