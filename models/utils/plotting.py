import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
import json

def bins_calc(lst: list, use: str) -> list:
    """
    Calculates the bins for a histrogram
    
    Parameters:
    ----------
    lst : list
        list of results from games
    use : str
        Choices of 'max' or 'game'
    """
    lst = sorted(list(set(lst)))
    largest = None
    if use == "max":
        largest = 2058
    elif use == "game":
        largest = int(lst[-1]) + 10
    else:
        raise ValueError("use: expexted max or game")

    l = [n for n in range(int(lst[0]))]
    r = [n for n in range(int(lst[-1]), largest)]
    return l + lst + r

def get_name(agent) -> str:
    """"
    Takes an agent object and returns the agents name.

    Parameters
    ----------
    agent: AgentObject
        An object: Agent{agent_name} representing an agent.
    """
    agent_name, agent_str = "", str(agent)
    ind = agent_str.find("Agent") + 5
    agent_str = agent_str.upper()
    for i in range(ind , len(agent_str)):
        if ord(agent_str[i]) > ord('Z') or ord(agent_str[i]) < ord('A'): break
        agent_name += agent_str[i]
    return agent_name

class Plotter():
    
    """
    A class for creating plots and saving information about the plots.
    """

    def __init__(self, game_scores=None, max_scores=None, num_steps=None, agent=None) -> None:
        """
        Initialize a Plotter object.

        Parameters
        ----------
        game_scores : dict
            A dictionary containing game scores.
        max_scores : dict
            A dictionary containing maximum scores.
        num_steps : dict
            A dictionary containing the number of steps.
        agent: AgentObject 
            An object representing an agent.
        """
        self.game_scores = game_scores
        self.max_scores = max_scores
        self.num_steps = num_steps
        self.num_episodes = sum(num_steps.values())
        self.agent_name = get_name(agent)
        self.my_path = os.path.dirname(os.path.dirname(__file__))

        self.json_path = os.path.join(os.path.dirname(__file__), 'data.json')

    def save_json_info(self):
        """
        Save information about the maximum and game scores, and the number of steps to a JSON file.
        """
        time = ''.join(c for c in str(datetime.now()) if c not in '.:')
        JSON_val = {
            "Agent Name" : f'{self.agent_name}',
            "# Episodes" : f'{self.num_episodes}',
            "Max Scores" : dict(sorted(self.max_scores.items())),
            "Game Scores": dict(sorted(self.game_scores.items())),
            "# of steps" : dict(sorted(self.num_steps.items()))
        }
        JSON_NAME = self.my_path + f'/data/json/{self.agent_name} plot_info' + time + '.json'
        with open(JSON_NAME, "w") as f:
            json.dump(JSON_val, f, indent=4)

        with open(JSON_NAME) as f:
            data = json.load(f)

        time = ''.join(c for c in str(datetime.now()) if c not in '.:')

        # Plot histogram of # of steps
        steps_data = data['# of steps']
        plt.hist(list(map(int, steps_data.keys())), bins=20)
        plt.title('Histogram of # of Steps')
        plt.xlabel('# of Steps')
        plt.ylabel('Frequency')
        outfile = self.my_path + f'/data/plots/{self.agent_name} NumSteps' + time + '.jpg'
        plt.savefig(outfile)
        plt.show()
        # Plot histogram of game scores
        scores_data = data['Game Scores']
        plt.hist(list(map(float, scores_data.keys())), bins=20)
        plt.title('Histogram of Game Scores')
        plt.xlabel('Game Score')
        plt.ylabel('Frequency')
        outfile = self.my_path + f'/data/plots/{self.agent_name} GameScore' + time + '.jpg'
        plt.savefig(outfile)
        plt.show()
        # Plot barchart of max scores
        max_scores_data = data['Max Scores']
        x_labels = [str(2**x) for x in range(12)]
        y_values = [max_scores_data.get(str(float(x)), 0) for x in x_labels]
        plt.bar(x_labels, y_values)
        plt.title('Max Scores')
        plt.xlabel('Tile Value')
        plt.ylabel('Frequency')
        outfile = self.my_path + f'/data/plots/{self.agent_name} MaxScore ' + time + '.jpg'
        plt.savefig(outfile)
        plt.show()

    
    # def plt_max_score(self, plt_type:str) -> None:
    #     """
    #     Create a plot of the max scores and save it to a file.
        
    #     Parameters
    #     ----------
    #     plt_type: str
    #         The parameters of choice are 'histo', 'bar'.
    #     """
    #     plt.clf()
    #     plt.title(f'Agent {self.agent_name} Max Game scores over {self.num_episodes} episodes')
    #     plt.xlabel("Max scores") 
    #     plt.ylabel("Frequency")

    #     if plt_type == "histo":
    #         max_scores = [key for key, val in self.max_scores.items() for _ in range(val)]
    #         plt.hist(max_scores, bins=bins_calc(lst=max_scores, use="max"), 
    #                 color="green")
    #     elif plt_type == "bar":
    #         plt.bar(list(self.max_scores.keys()), self.max_scores.values(),
    #                  color="green")
            
    #     time = ''.join(c for c in str(datetime.now()) if c not in '.:')
    #     outfile = self.my_path + f'/data/plots/{self.agent_name} MaxScore ' + time + '.jpg'
    #     plt.savefig(outfile)

    # def plt_game_score(self, plt_type:str) -> None:
    #     """
    #     Create a plot of the game scores and save it to a file.
        
    #     Parameters
    #     ----------
    #     plt_type: str
    #         The parameters of choice are 'histo', 'bar'.
    #     """
    #     plt.clf()
    #     plt.title(f'Agent {self.agent_name} Overall Game scores over {self.num_episodes} episodes')
    #     plt.xlabel("Game scores") 
    #     plt.ylabel("Frequency")

    #     if plt_type == "histo":
    #         game_scores = [key for key, val in self.game_scores.items() for _ in range(val)]
    #         plt.hist(game_scores, bins=bins_calc(game_scores, use="game"), 
    #                 color="green")
    #     elif plt_type == "bar":
    #         plt.bar(list(self.game_scores.keys()), self.game_scores.values(), 
    #                 color="green")
        
    #     time = ''.join(c for c in str(datetime.now()) if c not in '.:')
    #     outfile = self.my_path + f'/data/plots/{self.agent_name} GameScore' + time + '.jpg'
    #     plt.savefig(outfile)
    