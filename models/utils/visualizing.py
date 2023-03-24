import os
import matplotlib.pyplot as plt
import torch

class Visualizer():
    
    """
    A class for creating images of the game and making videos of the game.
    """

    def __init__(self, full_simulation_tensor=None, full_stats_tensor=None, best_game_tensor=None, best_stats_tensor=None) -> None:
        """
        Initialize a Visualizer object.

        Args:
 
               full_simulation_tensor: tensor with information from all episodes.
               full_stats_tensor: 
        """
        
        
        # Extract the name of the agent from its string representation.
        agent_name, agent_str = "", str(agent)
        for i in range(agent_str.find("Agent") + 5, len(agent_str)):
            if ord(agent_str[i]) == ord(' '): break
            agent_name += agent_str[i]
        
        self.agent_name = agent_name

        self.my_path = os.path.dirname(__file__)
    