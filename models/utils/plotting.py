import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
import json

def bins_calc(lst: list, use: str) -> list:
    lst = sorted(list(set(lst)))
    l = [n for n in range(int(lst[0]))]
    r = [n for n in range(int(lst[-1]), 2058 if use == "max" else int(lst[-1]) + 10,10)]
    return l + lst + r

class Plotter():
    def __init__(self, game_scores=None, max_scores=None, num_steps=None, agent=None) -> None:
        self.game_scores = game_scores
        self.max_scores = max_scores
        self.num_steps = num_steps
        
        agent_name, agent_str = "", str(agent)
        for i in range(agent_str.find("Agent") + 5, len(agent_str)):
            if ord(agent_str[i]) == ord(' '): break
            agent_name += agent_str[i]
        
        self.agent_name = agent_name

        self.my_path = os.path.dirname(__file__)
    
    def plt_max_score(self):
        plt.clf()
        max_scores = [key for key, val in self.max_scores.items() for _ in range(val)]
        plt.title(f'Agent: {self.agent_name} Max Game scores over n episodes')
        plt.xlabel("Max scores") 
        plt.ylabel("Frequency")

        plt.hist(max_scores, bins=bins_calc(lst=max_scores, use="max"), 
                 color="green")
        time = ''.join(c for c in str(datetime.now()) if c not in '.:')
        outfile = self.my_path + '/plots/MaxScore ' + time + '.jpg'
        plt.savefig(outfile)

    def plt_game_score(self):
        plt.clf()

        game_scores = [key for key, val in self.game_scores.items() for _ in range(val)]
        plt.title(f'Agent: {self.agent_name} Overall Game scores over n episodes')
        plt.xlabel("Game scores") 
        plt.ylabel("Frequency")
        plt.hist(game_scores, bins=bins_calc(game_scores, use="gs"), 
                 color="green")
        time = ''.join(c for c in str(datetime.now()) if c not in '.:')
        outfile = self.my_path + '/plots/GameScore ' + time + '.jpg'
        plt.savefig(outfile)

    def save_info(self):
        time = ''.join(c for c in str(datetime.now()) if c not in '.:')
        JSON_val = {
            "Max Scores" : dict(sorted(self.max_scores.items())),
            "Game Scores": dict(sorted(self.game_scores.items())),
            "# of steps" : dict(sorted(self.num_steps.items()))
        }
        with open(self.my_path + '\plots\plot_info' + time + '.json', "w") as f:
            json.dump(JSON_val, f, indent=4)
    