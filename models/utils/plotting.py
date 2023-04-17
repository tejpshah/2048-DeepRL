import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
import json

def get_name(agent) -> str:
    """"
    Takes an agent object and returns the agents name.

    Parameters
    ----------
    agent: AgentObject
        An object: Agent{agent_name} representing an agent.
    """
    # Extract the name of the agent from its string representation
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
        self.num_episodes = sum(num_steps.values()) # Calculate the total number of episodes played
        self.agent_name = get_name(agent) # Extract the agent name from the agent object
        self.my_path = os.path.dirname(os.path.dirname(__file__)) # Get the path to the project directory

    def save_json_info(self):
        """
        Save information about the maximum and game scores, and the number of steps to a JSON file.
        """
        # Create a dictionary with the necessary information
        JSON_val = {
            "Agent Name" : f'{self.agent_name}',
            "# Episodes" : f'{self.num_episodes}',
            "Max Scores" : dict(sorted(self.max_scores.items())),
            "Game Scores": dict(sorted(self.game_scores.items())),
            "# of steps" : dict(sorted(self.num_steps.items()))
        }

        # Generate a unique filename for the JSON file based on the current time
        time = ''.join(c for c in str(datetime.now()) if c not in '.:')
        JSON_name = self.agent_name + ' plot_info' + time + '.json'
        JSON_path = os.path.join(self.my_path, 'data', 'JSON', JSON_name)

        # Save the dictionary as a JSON file
        with open(JSON_path, "w") as f:
            json.dump(JSON_val, f, indent=4)

        # Load the JSON file back into memory
        with open(JSON_path) as f:
            data = json.load(f)

        # Plot histogram of # of steps
        steps_data = data['# of steps']
        plt.hist(list(map(int, steps_data.keys())), bins=20)
        plt.title('Histogram of # of Steps')
        plt.xlabel('# of Steps')
        plt.ylabel('Frequency')
        outfile = os.path.join(self.my_path, 'data', 'plots', f'{self.agent_name} NumSteps' + time + '.jpg')
        plt.savefig(outfile)
        plt.show()

        # Plot histogram of game scores
        scores_data = data['Game Scores']
        plt.hist(list(map(float, scores_data.keys())), bins=20)
        plt.title('Histogram of Game Scores')
        plt.xlabel('Game Score')
        plt.ylabel('Frequency')
        outfile = os.path.join(self.my_path, 'data', 'plots', f'{self.agent_name} GameScore' + time + '.jpg')
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
        outfile = os.path.join(self.my_path, 'data', 'plots', f'{self.agent_name} MaxScore' + time + '.jpg')
        plt.savefig(outfile)
        plt.show()