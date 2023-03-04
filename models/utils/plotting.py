import os
import matplotlib.pyplot as plt
import multiprocessing as mp

class Plotter():
    def __init__(self, game_scores=None, max_scores=None, num_steps=None, agent_name=None) -> None:
        self.game_scores = game_scores
        self.max_scores = max_scores
        self.num_steps = num_steps
        self.agent_name = agent_name

        self.my_path = os.path.dirname(__file__)

    def plt_max(self):
        plt.title(f'Agent: {self.agent_name} Max Game scores over n episodes')
        plt.xlabel("Max scores") 
        plt.ylabel("Frequency")
        plt.bar(self.max_scores.keys(),self.max_scores.values(), color='g')
        plt.savefig(self.my_path + '/plots/test.jpg')
        