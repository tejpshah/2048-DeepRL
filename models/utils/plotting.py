import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
import json

def bins_calc(lst, use):
    """
    Calculate the bins for a histogram.

    Args:
        lst: A list of scores.
        use: A string indicating whether the bins are for maximum or game scores.

    Returns:
        A list of bin edges.
    """
    lst = sorted(list(set(lst)))
    l = [n for n in range(int(lst[0]))]
    r = [n for n in range(int(lst[-1]), 2058 if use == "max" else int(lst[-1]) + 10,10)]
    return l + lst + r

class Plotter():
    
    """
    A class for creating plots and saving information about the plots.
    """

    def __init__(self, game_scores=None, max_scores=None, num_steps=None, agent=None) -> None:
        """
        Initialize a Plotter object.

        Args:
            game_scores: A dictionary containing game scores.
            max_scores: A dictionary containing maximum scores.
            num_steps: A dictionary containing the number of steps.
            agent: An object representing an agent.
        """
        self.game_scores = game_scores
        self.max_scores = max_scores
        self.num_steps = num_steps
        
        # Extract the name of the agent from its string representation.
        agent_name, agent_str = "", str(agent)
        for i in range(agent_str.find("Agent") + 5, len(agent_str)):
            if ord(agent_str[i]) == ord(' '): break
            agent_name += agent_str[i]
        
        self.agent_name = agent_name

        self.my_path = os.path.dirname(__file__)
    
    def plt_max_score(self):
        """
        Create a plot of the maximum scores and save it to a file.
        """
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
        """
        Create a plot of the game scores and save it to a file.
        """
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
        """
        Save information about the maximum and game scores, and the number of steps to a JSON file.
        """
        time = ''.join(c for c in str(datetime.now()) if c not in '.:')
        JSON_val = {
            "Max Scores" : dict(sorted(self.max_scores.items())),
            "Game Scores": dict(sorted(self.game_scores.items())),
            "# of steps" : dict(sorted(self.num_steps.items()))
        }
        with open(self.my_path + '\plots\plot_info' + time + '.json', "w") as f:
            json.dump(JSON_val, f, indent=4)
    